#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

__constant__ int kMidBand[30] = {
    3, 0, 2, 1, 1, 2, 0, 3, 4, 0,
    3, 1, 2, 2, 1, 3, 0, 4, 5, 0,
    4, 1, 3, 2, 2, 3, 1, 4, 0, 5
};

static inline void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

static inline char wm_bit_from_index(uint32_t idx, uint32_t key) {
    uint32_t x = idx ^ (key * 0x9E3779B9u);
    x ^= (x >> 16);
    x *= 0x7FEB352Du;
    x ^= (x >> 15);
    x *= 0x846CA68Bu;
    x ^= (x >> 16);
    return (x & 1u) ? 1 : -1;
}

__device__ inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline void bgr_to_ycrcb(float b, float g, float r, float* y, float* cr, float* cb) {
    float yy = 0.299f * r + 0.587f * g + 0.114f * b;
    float crr = (r - yy) * 0.713f + 128.0f;
    float cbb = (b - yy) * 0.564f + 128.0f;
    *y = yy;
    *cr = crr;
    *cb = cbb;
}

__device__ inline void ycrcb_to_bgr(float y, float cr, float cb, float* b, float* g, float* r) {
    float crr = cr - 128.0f;
    float cbb = cb - 128.0f;
    *r = y + 1.403f * crr;
    *b = y + 1.773f * cbb;
    *g = y - 0.714f * crr - 0.344f * cbb;
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

__device__ inline float idct_pixel(int x, int y, const float* dct) {
    constexpr float PI = 3.14159265358979323846f;
    float sum = 0.0f;
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            float au = (u == 0) ? 0.70710678f : 1.0f;
            float av = (v == 0) ? 0.70710678f : 1.0f;
            float cx = cosf(((2.0f * x + 1.0f) * static_cast<float>(u) * PI) / 16.0f);
            float cy = cosf(((2.0f * y + 1.0f) * static_cast<float>(v) * PI) / 16.0f);
            sum += au * av * dct[v * 8 + u] * cx * cy;
        }
    }
    return 0.25f * sum;
}

__global__ void watermark_block(
    const unsigned char* in_bgr,
    unsigned char* out_bgr,
    int width,
    int height,
    int blocks_w,
    const signed char* watermark,
    int mid_len,
    float alpha
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_x = gid % blocks_w;
    int block_y = gid / blocks_w;
    int base_x = block_x * 8;
    int base_y = block_y * 8;
    if (base_x + 7 >= width || base_y + 7 >= height) return;

    float yblock[64];
    float crblock[64];
    float cbblock[64];
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
            crblock[y * 8 + x] = cr;
            cbblock[y * 8 + x] = cb;
        }
    }

    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) {
            dct[v * 8 + u] = dct_coeff(u, v, yblock);
        }
    }

    int base_wm = gid * mid_len;
    for (int k = 0; k < mid_len; ++k) {
        int u = kMidBand[2 * k + 0];
        int v = kMidBand[2 * k + 1];
        float w = static_cast<float>(watermark[base_wm + k]);
        dct[v * 8 + u] += alpha * w;
    }

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float yy = idct_pixel(x, y, dct);
            float cr = crblock[y * 8 + x];
            float cb = cbblock[y * 8 + x];
            float b, g, r;
            ycrcb_to_bgr(yy, cr, cb, &b, &g, &r);
            int px = base_x + x;
            int py = base_y + y;
            int idx = (py * width + px) * 3;
            out_bgr[idx + 0] = static_cast<unsigned char>(clampf(b, 0.0f, 255.0f));
            out_bgr[idx + 1] = static_cast<unsigned char>(clampf(g, 0.0f, 255.0f));
            out_bgr[idx + 2] = static_cast<unsigned char>(clampf(r, 0.0f, 255.0f));
        }
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

__global__ void tpvm_apply(
    const unsigned char* in_bgr,
    unsigned char* out_a,
    unsigned char* out_b,
    int width,
    int height,
    float strength
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (gid >= total) return;

    int x = gid % width;
    int y = gid / width;
    float p = ((x + y) & 1) ? 1.0f : -1.0f;
    float delta = p * strength;

    int idx = gid * 3;
    float b = static_cast<float>(in_bgr[idx + 0]);
    float g = static_cast<float>(in_bgr[idx + 1]);
    float r = static_cast<float>(in_bgr[idx + 2]);

    out_a[idx + 0] = static_cast<unsigned char>(clampf(b + delta, 0.0f, 255.0f));
    out_a[idx + 1] = static_cast<unsigned char>(clampf(g + delta, 0.0f, 255.0f));
    out_a[idx + 2] = static_cast<unsigned char>(clampf(r + delta, 0.0f, 255.0f));

    out_b[idx + 0] = static_cast<unsigned char>(clampf(b - delta, 0.0f, 255.0f));
    out_b[idx + 1] = static_cast<unsigned char>(clampf(g - delta, 0.0f, 255.0f));
    out_b[idx + 2] = static_cast<unsigned char>(clampf(r - delta, 0.0f, 255.0f));
}

int main() {
    const char* input_video = "sample3.mp4";
    const char* output_video = "hybrid_protected_output3.mp4";

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        std::printf("Input video not found: %s\n", input_video);
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    int width_trim = (width / 8) * 8;
    int height_trim = (height / 8) * 8;
    if (width_trim != width || height_trim != height) {
        std::printf("Warning: frame size not multiple of 8, cropping to %dx%d\n", width_trim, height_trim);
    }

    cv::VideoWriter out(
        output_video,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps * 2.0,
        cv::Size(width_trim, height_trim));
    if (!out.isOpened()) {
        std::printf("Failed to open output video: %s\n", output_video);
        return 1;
    }

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

    int blocks_w = width_trim / 8;
    int blocks_h = height_trim / 8;
    int total_blocks = blocks_w * blocks_h;
    const int mid_len = 15;
    const float alpha = 10.0f;
    const float strength = 40.0f;
    const uint32_t watermark_key = 42;

    std::vector<signed char> watermark(static_cast<size_t>(total_blocks) * mid_len);
    for (uint32_t i = 0; i < watermark.size(); ++i) {
        watermark[i] = static_cast<signed char>(wm_bit_from_index(i, watermark_key));
    }

    size_t frame_bytes = static_cast<size_t>(width_trim) * static_cast<size_t>(height_trim) * 3;
    unsigned char* d_in = nullptr;
    unsigned char* d_wm = nullptr;
    unsigned char* d_out_a = nullptr;
    unsigned char* d_out_b = nullptr;
    signed char* d_wm_bits = nullptr;
    float* d_block_sums = nullptr;

    check_cuda(cudaMalloc(&d_in, frame_bytes), "cudaMalloc(d_in)");
    check_cuda(cudaMalloc(&d_wm, frame_bytes), "cudaMalloc(d_wm)");
    check_cuda(cudaMalloc(&d_out_a, frame_bytes), "cudaMalloc(d_out_a)");
    check_cuda(cudaMalloc(&d_out_b, frame_bytes), "cudaMalloc(d_out_b)");
    check_cuda(cudaMalloc(&d_wm_bits, watermark.size()), "cudaMalloc(d_wm_bits)");
    check_cuda(cudaMalloc(&d_block_sums, sizeof(float) * total_blocks), "cudaMalloc(d_block_sums)");
    check_cuda(cudaMemcpy(d_wm_bits, watermark.data(), watermark.size(), cudaMemcpyHostToDevice), "cudaMemcpy(wm_bits)");

    std::vector<float> sums(total_blocks);
    cv::Mat frame;
    cv::Mat frame_trim(height_trim, width_trim, CV_8UC3);
    cv::Mat out_frame_a(height_trim, width_trim, CV_8UC3);
    cv::Mat out_frame_b(height_trim, width_trim, CV_8UC3);

    int frame_count = 0;
    std::printf("Starting CUDA processing (Watermark + TPVM)...\n");

    int threads = 256;
    int blocks_for_blocks = (total_blocks + threads - 1) / threads;
    int total_pixels = width_trim * height_trim;
    int blocks_for_pixels = (total_pixels + threads - 1) / threads;

    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) break;

        if (frame.cols != width_trim || frame.rows != height_trim) {
            frame_trim = frame(cv::Rect(0, 0, width_trim, height_trim)).clone();
        } else {
            frame_trim = frame;
        }

        check_cuda(cudaMemcpy(d_in, frame_trim.data, frame_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(frame)");

        watermark_block<<<blocks_for_blocks, threads>>>(
            d_in, d_wm, width_trim, height_trim, blocks_w, d_wm_bits, mid_len, alpha);
        check_cuda(cudaGetLastError(), "watermark_block launch");

        tpvm_apply<<<blocks_for_pixels, threads>>>(
            d_wm, d_out_a, d_out_b, width_trim, height_trim, strength);
        check_cuda(cudaGetLastError(), "tpvm_apply launch");

        if (frame_count % 10 == 0) {
            detect_block<<<blocks_for_blocks, threads>>>(
                d_wm, width_trim, height_trim, blocks_w, d_wm_bits, mid_len, d_block_sums);
            check_cuda(cudaGetLastError(), "detect_block launch");
            check_cuda(cudaMemcpy(sums.data(), d_block_sums, sizeof(float) * total_blocks, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(block_sums)");
            double total = 0.0;
            for (int i = 0; i < total_blocks; ++i) total += sums[i];
            double corr = total / static_cast<double>(total_blocks * mid_len);
            std::printf("Frame %d: WM Correlation in Frame A = %.2f (Threshold > 2.0 usually)\n", frame_count, corr);
        }

        check_cuda(cudaMemcpy(out_frame_a.data, d_out_a, frame_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(out_a)");
        check_cuda(cudaMemcpy(out_frame_b.data, d_out_b, frame_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(out_b)");

        out.write(out_frame_a);
        out.write(out_frame_b);
        frame_count++;
    }

    cudaFree(d_in);
    cudaFree(d_wm);
    cudaFree(d_out_a);
    cudaFree(d_out_b);
    cudaFree(d_wm_bits);
    cudaFree(d_block_sums);
    cap.release();
    out.release();

    std::printf("Processing complete. Saved to %s\n", output_video);
    return 0;
}
