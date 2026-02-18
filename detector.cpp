#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>

static const char* kKernelSource = R"CLC(
__constant float PI = 3.14159265358979323846f;

inline void bgr_to_ycrcb(float b, float g, float r, float* y, float* cr, float* cb) {
    float yy = 0.299f * r + 0.587f * g + 0.114f * b;
    float crr = (r - yy) * 0.713f + 128.0f;
    float cbb = (b - yy) * 0.564f + 128.0f;
    *y = yy;
    *cr = crr;
    *cb = cbb;
}

inline float dct_coeff(int u, int v, __private float* block) {
    float sum = 0.0f;
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            float pixel = block[y * 8 + x];
            float cx = cos(((2.0f * x + 1.0f) * u * PI) / 16.0f);
            float cy = cos(((2.0f * y + 1.0f) * v * PI) / 16.0f);
            sum += pixel * cx * cy;
        }
    }
    float au = (u == 0) ? 0.70710678f : 1.0f;
    float av = (v == 0) ? 0.70710678f : 1.0f;
    return 0.25f * au * av * sum;
}

__kernel void detect_block(
    __global const uchar* in_bgr,
    int width,
    int height,
    int blocks_w,
    __global const char* watermark,
    int mid_len,
    __global float* block_sums
) {
    int gid = get_global_id(0);
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
            float b = (float)in_bgr[idx + 0];
            float g = (float)in_bgr[idx + 1];
            float r = (float)in_bgr[idx + 2];
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

    const int mid_band[15][2] = {
        {3,0}, {2,1}, {1,2}, {0,3},
        {4,0}, {3,1}, {2,2}, {1,3}, {0,4},
        {5,0}, {4,1}, {3,2}, {2,3}, {1,4}, {0,5}
    };

    int base_wm = gid * mid_len;
    float sum = 0.0f;
    for (int k = 0; k < mid_len; ++k) {
        int u = mid_band[k][0];
        int v = mid_band[k][1];
        float w = (float)watermark[base_wm + k];
        sum += dct[v * 8 + u] * w;
    }

    block_sums[gid] = sum;
}
)CLC";

static void check_cl(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        printf("OpenCL error %d: %s\n", err, msg);
        exit(1);
    }
}

static char wm_bit_from_index(uint32_t idx, uint32_t key) {
    uint32_t x = idx ^ (key * 0x9E3779B9u);
    x ^= (x >> 16);
    x *= 0x7FEB352Du;
    x ^= (x >> 15);
    x *= 0x846CA68Bu;
    x ^= (x >> 16);
    return (x & 1u) ? 1 : -1;
}

int main(int argc, char** argv) {
    const char* input_video = (argc > 1) ? argv[1] : "sample3.mp4";
    const double detect_threshold = 0.25;
    const double possible_threshold = 0.10;
    const int max_pairs = 10;

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Could not open input: %s\n", input_video);
        return 1;
    }

    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width_trim = (width / 8) * 8;
    int height_trim = (height / 8) * 8;

    cl_int err = CL_SUCCESS;
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, NULL, &platform_count), "clGetPlatformIDs(count)");
    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), NULL), "clGetPlatformIDs(list)");

    cl_device_id device = NULL;
    for (cl_uint i = 0; i < platform_count; ++i) {
        cl_uint dev_count = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
        if (dev_count == 0) continue;
        std::vector<cl_device_id> devs(dev_count);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, dev_count, devs.data(), NULL);
        device = devs[0];
        break;
    }
    if (!device) {
        printf("No GPU device found.\n");
        return 1;
    }

    char device_name[256] = {0};
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Using Device: %s\n", device_name);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_cl(err, "clCreateContext");
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    check_cl(err, "clCreateCommandQueue");

    const char* src = kKernelSource;
    size_t src_len = strlen(kKernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    check_cl(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        log[log_size] = 0;
        printf("Build log:\n%s\n", log.data());
        return 1;
    }

    cl_kernel k_detect = clCreateKernel(program, "detect_block", &err);
    check_cl(err, "clCreateKernel(detect_block)");

    int blocks_w = width_trim / 8;
    int blocks_h = height_trim / 8;
    int total_blocks = blocks_w * blocks_h;
    const int mid_len = 15;
    const uint32_t watermark_key = 42;

    std::vector<char> watermark(total_blocks * mid_len);
    for (uint32_t i = 0; i < watermark.size(); ++i) {
        watermark[i] = wm_bit_from_index(i, watermark_key);
    }

    size_t frame_bytes = (size_t)width_trim * (size_t)height_trim * 3;
    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, frame_bytes, NULL, &err);
    check_cl(err, "clCreateBuffer(in_buf)");
    cl_mem wm_bits = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, watermark.size(), watermark.data(), &err);
    check_cl(err, "clCreateBuffer(wm_bits)");
    cl_mem block_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * total_blocks, NULL, &err);
    check_cl(err, "clCreateBuffer(block_sums)");

    std::vector<float> sums(total_blocks);
    cv::Mat frame_a, frame_b;
    cv::Mat trim_a(height_trim, width_trim, CV_8UC3);
    cv::Mat trim_b(height_trim, width_trim, CV_8UC3);
    cv::Mat restored(height_trim, width_trim, CV_8UC3);

    printf("----------------------------------------\n");
    printf("%-10s | %-20s | %s\n", "Pair", "Correlation Score", "Verdict");
    printf("----------------------------------------\n");

    int pair_count = 0;
    int detected_count = 0;
    double total_score = 0.0;

    while (pair_count < max_pairs) {
        if (!cap.read(frame_a)) break;
        if (!cap.read(frame_b)) break;

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

        check_cl(clEnqueueWriteBuffer(queue, in_buf, CL_TRUE, 0, frame_bytes, restored.data, 0, NULL, NULL), "clEnqueueWriteBuffer(in_buf)");
        check_cl(clSetKernelArg(k_detect, 0, sizeof(cl_mem), &in_buf), "set k_detect arg0");
        check_cl(clSetKernelArg(k_detect, 1, sizeof(int), &width_trim), "set k_detect arg1");
        check_cl(clSetKernelArg(k_detect, 2, sizeof(int), &height_trim), "set k_detect arg2");
        check_cl(clSetKernelArg(k_detect, 3, sizeof(int), &blocks_w), "set k_detect arg3");
        check_cl(clSetKernelArg(k_detect, 4, sizeof(cl_mem), &wm_bits), "set k_detect arg4");
        check_cl(clSetKernelArg(k_detect, 5, sizeof(int), &mid_len), "set k_detect arg5");
        check_cl(clSetKernelArg(k_detect, 6, sizeof(cl_mem), &block_sums), "set k_detect arg6");

        size_t g_blocks = (size_t)total_blocks;
        check_cl(clEnqueueNDRangeKernel(queue, k_detect, 1, NULL, &g_blocks, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(k_detect)");
        check_cl(clEnqueueReadBuffer(queue, block_sums, CL_TRUE, 0, sizeof(float) * total_blocks, sums.data(), 0, NULL, NULL), "clEnqueueReadBuffer(block_sums)");

        double total = 0.0;
        for (int i = 0; i < total_blocks; ++i) total += sums[i];
        double corr = total / (double)(total_blocks * mid_len);

        const char* verdict = "NO MATCH";
        if (corr > detect_threshold) {
            verdict = "DETECTED";
            detected_count++;
        } else if (corr > possible_threshold) {
            verdict = "POSSIBLE";
        }

        printf("%-10d | %-20.4f | %s\n", pair_count, corr, verdict);
        total_score += corr;
        pair_count++;
    }

    printf("----------------------------------------\n");
    double avg = (pair_count > 0) ? (total_score / pair_count) : 0.0;
    printf("Analyzed %d TPVM frame-pairs.\n", pair_count);
    printf("Average Correlation Score: %.4f\n", avg);
    printf("Detected pairs (>%.2f): %d/%d\n", detect_threshold, detected_count, pair_count);
    if (avg > detect_threshold) {
        printf(">>> RESULT: WATERMARK CONFIRMED (PIRATED COPY DETECTED) <<<\n");
    } else {
        printf(">>> RESULT: CLEAN / NO WATERMARK DETECTED <<<\n");
    }

    clReleaseMemObject(in_buf);
    clReleaseMemObject(wm_bits);
    clReleaseMemObject(block_sums);
    clReleaseKernel(k_detect);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    cap.release();
    return 0;
}
