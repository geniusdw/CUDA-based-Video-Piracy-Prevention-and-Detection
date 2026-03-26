#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <vector>
#include <filesystem>
#include <sstream>
#include "watermark_common.h"

// ==========================
// User-tunable calibration parameters
// ==========================
constexpr const char* kDefaultInputVideo = "test.mp4";            // Input video path used when no CLI arg is passed.
constexpr const char* kDefaultOutputVideo = "test_output.mp4"; // Output video path used when no CLI arg is passed.
constexpr float kOutputFpsMultiplier = 4.8f;                        // 25 fps source -> 120 fps output on average.

// High-Frequency Signal Parameters
constexpr float kLumaAmplitude = 38.0f;         // Lower spatial frequency survives weak optics, so we can push harder.
constexpr float kLumaSpatialFreq = 16.0f;       // Broader bands are more likely to survive webcam lens/sensor MTF.
constexpr float kChromaAmplitude = 24.0f;       // Checkerboard chroma still targets Bayer/demosaicing weaknesses.
constexpr float kRollingShutterRowTime = 46e-6f; // Approximate FRONTECH row readout time at 720p/30 fps.
constexpr float kDctEmbedStrength = 3.0f;       // Strength of the forensic DCT watermark embedded in both A/B frames.
constexpr const char* kWatermarkPayload = "Toshit's laptop";

constexpr int kCudaThreads = 256;                                   // CUDA threads per block for all kernels.
__constant__ int kMidBand[30];

static inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static inline void decode_fourcc(int fourcc, char out[5]) {
    out[0] = static_cast<char>(fourcc & 0xFF);
    out[1] = static_cast<char>((fourcc >> 8) & 0xFF);
    out[2] = static_cast<char>((fourcc >> 16) & 0xFF);
    out[3] = static_cast<char>((fourcc >> 24) & 0xFF);
    out[4] = '\0';

    for (int i = 0; i < 4; ++i) {
        if (out[i] == '\0') out[i] = ' ';
    }
}

static inline int frames_for_source_frame(int source_frame_index) {
    double start = std::floor(static_cast<double>(source_frame_index) * static_cast<double>(kOutputFpsMultiplier) + 1e-9);
    double end = std::floor(static_cast<double>(source_frame_index + 1) * static_cast<double>(kOutputFpsMultiplier) + 1e-9);
    return static_cast<int>(end - start);
}

static std::string build_full_payload() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
    localtime_s(&local_tm, &now_time);

    std::ostringstream payload_stream;
    payload_stream << kWatermarkPayload << ' ' << std::put_time(&local_tm, "%Y-%m-%d %H:%M");
    return payload_stream.str();
}

__device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline float dct_basis(int coord, int freq) {
    constexpr float kPi = 3.14159265358979323846f;
    return cosf(((2.0f * static_cast<float>(coord) + 1.0f) * static_cast<float>(freq) * kPi) / 16.0f);
}

// Struct for passing calibration parameters to the kernel
struct CalibrationParams {
    float luma_amplitude;
    float luma_spatial_freq_ky;
    float chroma_amplitude;
    float rolling_shutter_row_time;
    float frame_time;
};

__global__ void processSignalInterference(
    const unsigned char* in_bgr,
    unsigned char* out_a,
    unsigned char* out_b,
    int width,
    int height,
    CalibrationParams params
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (gid >= total_pixels) return;

    int x = gid % width;
    int y = gid / width;

    constexpr float kTwoPi = 6.28318530717958647692f;
    float row_phase = static_cast<float>(y) * params.rolling_shutter_row_time / params.frame_time;
    float rolling_offset = kTwoPi * row_phase;
    float spatial = params.luma_spatial_freq_ky * kTwoPi * static_cast<float>(y) / static_cast<float>(height);
    float luma_delta = params.luma_amplitude * __sinf(spatial + rolling_offset);

    // Checkerboard chroma averages out for the eye but interacts strongly with Bayer demosaicing.
    float checker = (((x + y) & 1) == 0) ? 1.0f : -1.0f;
    float chroma_r_delta = params.chroma_amplitude * checker;
    float chroma_b_delta = -params.chroma_amplitude * checker;

    int idx = gid * 3;
    float b = static_cast<float>(in_bgr[idx + 0]);
    float g = static_cast<float>(in_bgr[idx + 1]);
    float r = static_cast<float>(in_bgr[idx + 2]);

    // A: +delta, B: -delta. At 120 fps, the eye integrates them toward the original.
    // The rolling-shutter row offset is there to make captured rows fall on different phases.
    out_a[idx + 0] = static_cast<unsigned char>(clampf(b + luma_delta + chroma_b_delta, 0.0f, 255.0f));
    out_a[idx + 1] = static_cast<unsigned char>(clampf(g + luma_delta, 0.0f, 255.0f));
    out_a[idx + 2] = static_cast<unsigned char>(clampf(r + luma_delta + chroma_r_delta, 0.0f, 255.0f));

    out_b[idx + 0] = static_cast<unsigned char>(clampf(b - luma_delta - chroma_b_delta, 0.0f, 255.0f));
    out_b[idx + 1] = static_cast<unsigned char>(clampf(g - luma_delta, 0.0f, 255.0f));
    out_b[idx + 2] = static_cast<unsigned char>(clampf(r - luma_delta - chroma_r_delta, 0.0f, 255.0f));
}

__global__ void embedDctWatermark(
    unsigned char* frame_bgr,
    int width,
    int height,
    int blocks_w,
    int blocks_h,
    const signed char* watermark_bits,
    int mid_len,
    float embed_strength
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = blocks_w * blocks_h;
    if (gid >= total_blocks) return;

    int block_x = gid % blocks_w;
    int block_y = gid / blocks_w;
    int base_x = block_x * 8;
    int base_y = block_y * 8;
    if (base_x + 7 >= width || base_y + 7 >= height) return;

    int base_wm = gid * mid_len;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float delta = 0.0f;
            for (int k = 0; k < mid_len; ++k) {
                int u = kMidBand[2 * k + 0];
                int v = kMidBand[2 * k + 1];
                float wm = static_cast<float>(watermark_bits[base_wm + k]);
                delta += wm * embed_strength * dct_basis(x, u) * dct_basis(y, v);
            }

            int idx = ((base_y + y) * width + (base_x + x)) * 3;
            float b = static_cast<float>(frame_bgr[idx + 0]);
            float g = static_cast<float>(frame_bgr[idx + 1]);
            float r = static_cast<float>(frame_bgr[idx + 2]);
            frame_bgr[idx + 0] = static_cast<unsigned char>(clampf(b + delta, 0.0f, 255.0f));
            frame_bgr[idx + 1] = static_cast<unsigned char>(clampf(g + delta, 0.0f, 255.0f));
            frame_bgr[idx + 2] = static_cast<unsigned char>(clampf(r + delta, 0.0f, 255.0f));
        }
    }
}

int main(int argc, char** argv) {
    const char* input_video = (argc > 1) ? argv[1] : kDefaultInputVideo;
    const char* output_video = (argc > 2) ? argv[2] : kDefaultOutputVideo;
    const std::string full_payload = build_full_payload();
    const std::filesystem::path input_path = std::filesystem::absolute(input_video);
    const std::filesystem::path output_path = std::filesystem::absolute(output_video);

    std::printf("Embedding payload: %s\n", full_payload.c_str());
    std::printf("Input video path: %s\n", input_path.string().c_str());
    std::printf("Output video path: %s\n", output_path.string().c_str());
    std::printf("Change paths by passing CLI args or editing kDefaultInputVideo / kDefaultOutputVideo.\n");

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::printf("Could not open input video: %s\n", input_video);
        std::printf("Current working directory: %s\n", std::filesystem::current_path().string().c_str());
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_count_estimate = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int fourcc_code = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    char source_codec[5];
    decode_fourcc(fourcc_code, source_codec);

    std::printf("Source codec: %s\n", source_codec);
    std::printf("Frame count: %d\n", frame_count_estimate);
    std::printf("Input FPS from metadata: %.2f\n", fps);

    if (fps <= 0.0 || fps > 240.0) {
        std::printf("WARNING: invalid FPS %.2f, forcing to 25.00\n", fps);
        fps = 25.0;
    }

    int width_trim = (width / 8) * 8;
    int height_trim = (height / 8) * 8;
    if (width_trim != width || height_trim != height) {
        std::printf("Warning: frame size not multiple of 8, cropping to %dx%d\n", width_trim, height_trim);
    }

    double output_fps = fps * static_cast<double>(kOutputFpsMultiplier);
    cv::VideoWriter out(
        output_video,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        output_fps,
        cv::Size(width_trim, height_trim));
    if (!out.isOpened()) {
        std::printf("Failed to open output video: %s\n", output_video);
        return 1;
    }
    std::printf("VideoWriter opened: %s\n", out.isOpened() ? "YES" : "NO");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::printf("No CUDA GPU found. Exiting.\n");
        return 1;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Using Device: %s\n", prop.name);
    std::printf("Input FPS: %.2f, Output FPS: %.2f\n", fps, output_fps);
    std::printf("Rolling shutter row time: %.2f us\n", static_cast<double>(kRollingShutterRowTime) * 1.0e6);
    check_cuda(cudaMemcpyToSymbol(kMidBand, kMidBandTable, sizeof(kMidBandTable)), "cudaMemcpyToSymbol(kMidBand)");

    int blocks_w = width_trim / 8;
    int blocks_h = height_trim / 8;
    int total_blocks = blocks_w * blocks_h;
    const int mid_len = kDctMidBandLength;
    const uint32_t watermark_key = kWatermarkKey;

    std::vector<signed char> payload_bits;
    encode_payload(full_payload, payload_bits);
    std::vector<signed char> watermark(static_cast<size_t>(total_blocks) * mid_len);
    for (std::size_t i = 0; i < watermark.size(); ++i) {
        watermark[i] = payload_bit_for_slot(i, payload_bits, watermark_key);
    }

    // === Memory Allocation ===
    size_t frame_bytes = static_cast<size_t>(width_trim) * static_cast<size_t>(height_trim) * 3;
    unsigned char *h_in = nullptr, *h_out_a = nullptr, *h_out_b = nullptr;
    check_cuda(cudaHostAlloc(&h_in, frame_bytes, cudaHostAllocDefault), "cudaHostAlloc(h_in)");
    check_cuda(cudaHostAlloc(&h_out_a, frame_bytes, cudaHostAllocDefault), "cudaHostAlloc(h_out_a)");
    check_cuda(cudaHostAlloc(&h_out_b, frame_bytes, cudaHostAllocDefault), "cudaHostAlloc(h_out_b)");

    unsigned char *d_in = nullptr, *d_out_a = nullptr, *d_out_b = nullptr;
    signed char* d_wm_bits = nullptr;
    check_cuda(cudaMalloc(&d_in, frame_bytes), "cudaMalloc(d_in)");
    check_cuda(cudaMalloc(&d_out_a, frame_bytes), "cudaMalloc(d_out_a)");
    check_cuda(cudaMalloc(&d_out_b, frame_bytes), "cudaMalloc(d_out_b)");
    check_cuda(cudaMalloc(&d_wm_bits, watermark.size()), "cudaMalloc(d_wm_bits)");
    check_cuda(cudaMemcpy(d_wm_bits, watermark.data(), watermark.size(), cudaMemcpyHostToDevice), "cudaMemcpy(d_wm_bits)");

    cv::Mat frame_in_pinned(height_trim, width_trim, CV_8UC3, h_in);
    cv::Mat frame_out_a_pinned(height_trim, width_trim, CV_8UC3, h_out_a);
    cv::Mat frame_out_b_pinned(height_trim, width_trim, CV_8UC3, h_out_b);

    cv::Mat frame;

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    int frame_count = 0;
    int total_output_frames_written = 0;
    std::printf("Starting CUDA processing for static A/B generation...\n");

    int threads = kCudaThreads;
    int total_pixels = width_trim * height_trim;
    int blocks_for_pixels = (total_pixels + threads - 1) / threads;
    int blocks_for_blocks = (total_blocks + threads - 1) / threads;

    while (true) {
        if (!cap.read(frame)) {
            std::printf("cap.read failed at frame %d\n", frame_count);
            break;
        }

        // frame_in_pinned wraps h_in, so the ROI copy writes directly into the pinned buffer.
        if (frame.cols != width_trim || frame.rows != height_trim) {
            cv::Mat roi = frame(cv::Rect(0, 0, width_trim, height_trim));
            roi.copyTo(frame_in_pinned);
        } else {
            frame.copyTo(frame_in_pinned);
        }

        CalibrationParams params;
        params.luma_amplitude = kLumaAmplitude;
        params.luma_spatial_freq_ky = kLumaSpatialFreq;
        params.chroma_amplitude = kChromaAmplitude;
        params.rolling_shutter_row_time = kRollingShutterRowTime;
        params.frame_time = 1.0f / static_cast<float>(output_fps);

        check_cuda(cudaMemcpyAsync(d_in, h_in, frame_bytes, cudaMemcpyHostToDevice, stream), "HtoD copy");

        processSignalInterference<<<blocks_for_pixels, threads, 0, stream>>>(
            d_in, d_out_a, d_out_b, width_trim, height_trim, params);
        check_cuda(cudaGetLastError(), "processSignalInterference launch");

        embedDctWatermark<<<blocks_for_blocks, threads, 0, stream>>>(
            d_out_a, width_trim, height_trim, blocks_w, blocks_h, d_wm_bits, mid_len, kDctEmbedStrength);
        check_cuda(cudaGetLastError(), "embedDctWatermark launch A");
        embedDctWatermark<<<blocks_for_blocks, threads, 0, stream>>>(
            d_out_b, width_trim, height_trim, blocks_w, blocks_h, d_wm_bits, mid_len, kDctEmbedStrength);
        check_cuda(cudaGetLastError(), "embedDctWatermark launch B");

        check_cuda(cudaMemcpyAsync(h_out_a, d_out_a, frame_bytes, cudaMemcpyDeviceToHost, stream), "DtoH copy A");
        check_cuda(cudaMemcpyAsync(h_out_b, d_out_b, frame_bytes, cudaMemcpyDeviceToHost, stream), "DtoH copy B");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        int output_frames_for_source = frames_for_source_frame(frame_count);
        if (output_frames_for_source <= 0) {
            std::printf("WARNING: source frame %d produced %d output frames, forcing 1\n", frame_count, output_frames_for_source);
            output_frames_for_source = 1;
        }
        for (int f = 0; f < output_frames_for_source; ++f) {
            bool use_a = ((total_output_frames_written + f) % 2) == 0;
            out.write(use_a ? frame_out_a_pinned : frame_out_b_pinned);
        }
        total_output_frames_written += output_frames_for_source;

        frame_count++;
        if (frame_count % 10 == 0) {
            std::printf("Processed %d source frames...\n", frame_count);
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    cudaFree(d_wm_bits);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out_a);
    cudaFreeHost(h_out_b);
    cap.release();
    out.release();

    std::printf("\nProcessing complete. Processed %d frames and wrote %d output frames. Saved to %s\n",
                frame_count,
                total_output_frames_written,
                output_path.string().c_str());
    return 0;
}
