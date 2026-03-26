#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <filesystem>
#include "watermark_common.h"

// ==========================
// User-tunable global parameters
// ==========================
constexpr const char* kDefaultInputVideo = "hybrid_protected_output.mp4"; // Video to analyze when no CLI input is provided.
constexpr double kDetectThreshold = 0.25;                                  // Correlation threshold for definite watermark detection.
constexpr double kPossibleThreshold = 0.10;                                // Correlation threshold for weak/possible detection.
constexpr int kFallbackMaxPairs = 1000000000;                              // Used when frame-count metadata is unavailable.
constexpr int kCudaThreads = 256;                                           // CUDA threads per block.
constexpr double kMinRatioShort = 0.10;                                     // Min detected ratio for short clips (<40 pairs).
constexpr double kMinRatioMedium = 0.05;                                    // Min detected ratio for medium clips (40-99 pairs).
constexpr double kMinRatioLong = 0.02;                                      // Min detected ratio for long clips (>=100 pairs).

__constant__ int kMidBand[30];

static inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static bool read_restored_pair(
    cv::VideoCapture& cap,
    int width_trim,
    int height_trim,
    cv::Mat& frame_a,
    cv::Mat& frame_b,
    cv::Mat& trim_a,
    cv::Mat& trim_b,
    cv::Mat& restored
) {
    if (!cap.read(frame_a)) return false;
    if (!cap.read(frame_b)) return false;

    if (frame_a.cols != width_trim || frame_a.rows != height_trim) {
        trim_a = frame_a(cv::Rect(0, 0, width_trim, height_trim)).clone();
    } else {
        trim_a = frame_a;
    }
    if (frame_b.cols != width_trim || frame_b.rows != height_trim) {
        trim_b = frame_b(cv::Rect(0, 0, width_trim, height_trim)).clone();
    } else {
        trim_b = frame_b;
    }

    cv::addWeighted(trim_a, 0.5, trim_b, 0.5, 0.0, restored);
    return true;
}

__device__ inline void bgr_to_ycrcb(float b, float g, float r, float* y, float* cr, float* cb) {
    float yy = 0.299f * r + 0.587f * g + 0.114f * b;
    float crr = (r - yy) * 0.713f + 128.0f;
    float cbb = (b - yy) * 0.564f + 128.0f;
    *y = yy;
    *cr = crr;
    *cb = cbb;
}

__device__ inline float dct_coeff(int u, int v, const float* block) {
    constexpr float PI = 3.14159265358979323846f;
    float sum = 0.0f;
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            float pixel = block[y * 8 + x];
            float cx = cosf(((2.0f * x + 1.0f) * static_cast<float>(u) * PI) / 16.0f);
            float cy = cosf(((2.0f * y + 1.0f) * static_cast<float>(v) * PI) / 16.0f);
            sum += pixel * cx * cy;
        }
    }
    float au = (u == 0) ? 0.70710678f : 1.0f;
    float av = (v == 0) ? 0.70710678f : 1.0f;
    return 0.25f * au * av * sum;
}

__global__ void extract_midband_scores(
    const unsigned char* in_bgr,
    int width,
    int height,
    int blocks_w,
    int total_blocks,
    int mid_len,
    float* slot_scores
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_blocks) return;

    int block_x = gid % blocks_w;
    int block_y = gid / blocks_w;
    int base_x = block_x * 8;
    int base_y = block_y * 8;
    if (base_x + 7 >= width || base_y + 7 >= height) return;

    float yblock[64];
    float dct[64];

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            int px = base_x + x;
            int py = base_y + y;
            int idx = (py * width + px) * 3;
            float b = static_cast<float>(in_bgr[idx + 0]);
            float g = static_cast<float>(in_bgr[idx + 1]);
            float r = static_cast<float>(in_bgr[idx + 2]);
            float yy, cr, cb;
            bgr_to_ycrcb(b, g, r, &yy, &cr, &cb);
            yblock[y * 8 + x] = yy;
        }
    }

    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            dct[v * 8 + u] = dct_coeff(u, v, yblock);
        }
    }

    int base_out = gid * mid_len;
    for (int k = 0; k < mid_len; ++k) {
        int u = kMidBand[2 * k + 0];
        int v = kMidBand[2 * k + 1];
        slot_scores[base_out + k] = dct[v * 8 + u];
    }
}

__global__ void detect_block(
    const unsigned char* in_bgr,
    int width,
    int height,
    int blocks_w,
    const signed char* watermark,
    int mid_len,
    float* block_sums
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_x = gid % blocks_w;
    int block_y = gid / blocks_w;
    int base_x = block_x * 8;
    int base_y = block_y * 8;
    if (base_x + 7 >= width || base_y + 7 >= height) return;

    float yblock[64];
    float dct[64];

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            int px = base_x + x;
            int py = base_y + y;
            int idx = (py * width + px) * 3;
            float b = static_cast<float>(in_bgr[idx + 0]);
            float g = static_cast<float>(in_bgr[idx + 1]);
            float r = static_cast<float>(in_bgr[idx + 2]);
            float yy, cr, cb;
            bgr_to_ycrcb(b, g, r, &yy, &cr, &cb);
            yblock[y * 8 + x] = yy;
        }
    }

    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            dct[v * 8 + u] = dct_coeff(u, v, yblock);
        }
    }

    int base_wm = gid * mid_len;
    float sum = 0.0f;
    for (int k = 0; k < mid_len; ++k) {
        int u = kMidBand[2 * k + 0];
        int v = kMidBand[2 * k + 1];
        float w = static_cast<float>(watermark[base_wm + k]);
        sum += dct[v * 8 + u] * w;
    }
    block_sums[gid] = sum;
}

int main(int argc, char** argv) {
    const char* input_video = (argc > 1) ? argv[1] : kDefaultInputVideo;
    const double detect_threshold = kDetectThreshold;
    const double possible_threshold = kPossibleThreshold;
    const int user_max_pairs = (argc > 2) ? std::atoi(argv[2]) : -1;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::printf("Could not open input: %s\n", input_video);
        std::printf("Current working directory: %s\n", std::filesystem::current_path().string().c_str());
        return 1;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames_est = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width_trim = (width / 8) * 8;
    int height_trim = (height / 8) * 8;
    int dynamic_max_pairs = (total_frames_est > 1) ? (total_frames_est / 2) : kFallbackMaxPairs;
    int max_pairs = (user_max_pairs > 0) ? user_max_pairs : dynamic_max_pairs;

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
        std::printf("No CUDA GPU found.\n");
        return 1;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    std::printf("Using Device: %s\n", prop.name);
    check_cuda(cudaMemcpyToSymbol(kMidBand, kMidBandTable, sizeof(kMidBandTable)), "cudaMemcpyToSymbol(kMidBand)");

    int blocks_w = width_trim / 8;
    int blocks_h = height_trim / 8;
    int total_blocks = blocks_w * blocks_h;
    const int mid_len = kDctMidBandLength;
    const uint32_t watermark_key = kWatermarkKey;
    const std::size_t payload_bits_length = kPayloadBitCount;
    const std::size_t watermark_slot_count = static_cast<std::size_t>(total_blocks) * static_cast<std::size_t>(mid_len);

    size_t frame_bytes = static_cast<size_t>(width_trim) * static_cast<size_t>(height_trim) * 3;
    unsigned char* d_in = nullptr;
    signed char* d_wm_bits = nullptr;
    float* d_block_sums = nullptr;
    float* d_slot_scores = nullptr;

    check_cuda(cudaMalloc(&d_in, frame_bytes), "cudaMalloc(d_in)");
    check_cuda(cudaMalloc(&d_wm_bits, watermark_slot_count), "cudaMalloc(d_wm_bits)");
    check_cuda(cudaMalloc(&d_block_sums, sizeof(float) * total_blocks), "cudaMalloc(d_block_sums)");
    check_cuda(cudaMalloc(&d_slot_scores, sizeof(float) * watermark_slot_count), "cudaMalloc(d_slot_scores)");

    std::vector<float> sums(total_blocks);
    std::vector<float> slot_scores(watermark_slot_count);
    std::vector<float> soft_bits(payload_bits_length, 0.0f);
    cv::Mat frame_a, frame_b;
    cv::Mat trim_a(height_trim, width_trim, CV_8UC3);
    cv::Mat trim_b(height_trim, width_trim, CV_8UC3);
    cv::Mat restored(height_trim, width_trim, CV_8UC3);

    int pair_count = 0;
    int threads = kCudaThreads;
    int blocks_for_blocks = (total_blocks + threads - 1) / threads;

    while (pair_count < max_pairs) {
        if (!read_restored_pair(cap, width_trim, height_trim, frame_a, frame_b, trim_a, trim_b, restored)) break;

        check_cuda(cudaMemcpy(d_in, restored.data, frame_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(restored)");
        extract_midband_scores<<<blocks_for_blocks, threads>>>(d_in, width_trim, height_trim, blocks_w, total_blocks, mid_len, d_slot_scores);
        check_cuda(cudaGetLastError(), "extract_midband_scores launch");
        check_cuda(cudaMemcpy(slot_scores.data(), d_slot_scores, sizeof(float) * watermark_slot_count, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(slot_scores)");

        for (std::size_t slot_index = 0; slot_index < watermark_slot_count; ++slot_index) {
            const std::size_t payload_index = payload_index_for_slot(slot_index, payload_bits_length, watermark_key);
            soft_bits[payload_index] += slot_scores[slot_index];
        }
        pair_count++;
    }

    const std::string recovered_payload = decode_payload(soft_bits);
    if (!recovered_payload.empty()) {
        std::printf(">>> EMBEDDED PAYLOAD: %s <\n", recovered_payload.c_str());
    } else {
        std::printf(">>> NO PAYLOAD RECOVERED (clean video or wrong key) <\n");
    }

    if (recovered_payload.empty()) {
        std::printf("Analyzed %d TPVM frame-pairs.\n", pair_count);
        std::printf(">>> RESULT: CLEAN / NO WATERMARK DETECTED <<<\n");
        cudaFree(d_in);
        cudaFree(d_wm_bits);
        cudaFree(d_block_sums);
        cudaFree(d_slot_scores);
        cap.release();
        return 0;
    }

    std::vector<signed char> payload_bits;
    encode_payload(recovered_payload, payload_bits);
    std::vector<signed char> watermark(watermark_slot_count);
    for (std::size_t i = 0; i < watermark.size(); ++i) {
        watermark[i] = payload_bit_for_slot(i, payload_bits, watermark_key);
    }
    check_cuda(cudaMemcpy(d_wm_bits, watermark.data(), watermark.size(), cudaMemcpyHostToDevice), "cudaMemcpy(wm_bits)");

    cap.release();
    cap.open(input_video);
    if (!cap.isOpened()) {
        std::printf("Could not reopen input: %s\n", input_video);
        cudaFree(d_in);
        cudaFree(d_wm_bits);
        cudaFree(d_block_sums);
        cudaFree(d_slot_scores);
        return 1;
    }

    std::printf("----------------------------------------\n");
    std::printf("%-10s | %-20s | %s\n", "Pair", "Correlation Score", "Verdict");
    std::printf("----------------------------------------\n");

    pair_count = 0;
    int detected_count = 0;
    double total_score = 0.0;

    while (pair_count < max_pairs) {
        if (!read_restored_pair(cap, width_trim, height_trim, frame_a, frame_b, trim_a, trim_b, restored)) break;

        check_cuda(cudaMemcpy(d_in, restored.data, frame_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(restored)");
        detect_block<<<blocks_for_blocks, threads>>>(d_in, width_trim, height_trim, blocks_w, d_wm_bits, mid_len, d_block_sums);
        check_cuda(cudaGetLastError(), "detect_block launch");
        check_cuda(cudaMemcpy(sums.data(), d_block_sums, sizeof(float) * total_blocks, cudaMemcpyDeviceToHost), "cudaMemcpy(sums)");

        double total = 0.0;
        for (int i = 0; i < total_blocks; ++i) total += sums[i];
        double corr = total / static_cast<double>(total_blocks * mid_len);

        const char* verdict = "NO MATCH";
        if (corr > detect_threshold) {
            verdict = "DETECTED";
            detected_count++;
        } else if (corr > possible_threshold) {
            verdict = "POSSIBLE";
        }

        std::printf("%-10d | %-20.4f | %s\n", pair_count, corr, verdict);
        total_score += corr;
        pair_count++;
    }

    std::printf("----------------------------------------\n");
    double avg = (pair_count > 0) ? (total_score / pair_count) : 0.0;
    double detected_ratio = (pair_count > 0) ? (static_cast<double>(detected_count) / pair_count) : 0.0;
    double min_ratio_for_confirm = kMinRatioShort;
    if (pair_count >= 40) min_ratio_for_confirm = kMinRatioMedium;
    if (pair_count >= 100) min_ratio_for_confirm = kMinRatioLong;
    int min_detected_pairs_for_confirm = static_cast<int>(std::ceil(pair_count * min_ratio_for_confirm));
    if (min_detected_pairs_for_confirm < 1) min_detected_pairs_for_confirm = 1;

    std::printf("Analyzed %d TPVM frame-pairs.\n", pair_count);
    std::printf("Average Correlation Score: %.4f\n", avg);
    std::printf("Detected pairs (>%.2f): %d/%d\n", detect_threshold, detected_count, pair_count);
    std::printf("Detected ratio: %.4f (min ratio for this length: %.4f)\n", detected_ratio, min_ratio_for_confirm);
    std::printf("Min detected pairs required for confirm: %d\n", min_detected_pairs_for_confirm);
    if (detected_count >= min_detected_pairs_for_confirm || avg > detect_threshold) {
        std::printf(">>> RESULT: WATERMARK CONFIRMED (PIRATED COPY DETECTED) <<<\n");
    } else {
        std::printf(">>> RESULT: CLEAN / NO WATERMARK DETECTED <<<\n");
    }

    cudaFree(d_in);
    cudaFree(d_wm_bits);
    cudaFree(d_block_sums);
    cudaFree(d_slot_scores);
    cap.release();
    return 0;
}
