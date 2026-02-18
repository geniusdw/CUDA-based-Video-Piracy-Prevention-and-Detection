#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <stdint.h>

static const char* kKernelSource = R"CLC(
__constant float PI = 3.14159265358979323846f;

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline void bgr_to_ycrcb(float b, float g, float r, float* y, float* cr, float* cb) {
    float yy = 0.299f * r + 0.587f * g + 0.114f * b;
    float crr = (r - yy) * 0.713f + 128.0f;
    float cbb = (b - yy) * 0.564f + 128.0f;
    *y = yy;
    *cr = crr;
    *cb = cbb;
}

inline void ycrcb_to_bgr(float y, float cr, float cb, float* b, float* g, float* r) {
    float crr = cr - 128.0f;
    float cbb = cb - 128.0f;
    float rr = y + 1.403f * crr;
    float bb = y + 1.773f * cbb;
    float gg = y - 0.714f * crr - 0.344f * cbb;
    *r = rr;
    *g = gg;
    *b = bb;
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

inline float idct_pixel(int x, int y, __private float* dct) {
    float sum = 0.0f;
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            float au = (u == 0) ? 0.70710678f : 1.0f;
            float av = (v == 0) ? 0.70710678f : 1.0f;
            float cx = cos(((2.0f * x + 1.0f) * u * PI) / 16.0f);
            float cy = cos(((2.0f * y + 1.0f) * v * PI) / 16.0f);
            sum += au * av * dct[v * 8 + u] * cx * cy;
        }
    }
    return 0.25f * sum;
}

__kernel void watermark_block(
    __global const uchar* in_bgr,
    __global uchar* out_bgr,
    int width,
    int height,
    int blocks_w,
    __global const char* watermark,
    int mid_len,
    float alpha
) {
    int gid = get_global_id(0);
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
            float b = (float)in_bgr[idx + 0];
            float g = (float)in_bgr[idx + 1];
            float r = (float)in_bgr[idx + 2];
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

    const int mid_band[15][2] = {
        {3,0}, {2,1}, {1,2}, {0,3},
        {4,0}, {3,1}, {2,2}, {1,3}, {0,4},
        {5,0}, {4,1}, {3,2}, {2,3}, {1,4}, {0,5}
    };

    int base_wm = gid * mid_len;
    for (int k = 0; k < mid_len; ++k) {
        int u = mid_band[k][0];
        int v = mid_band[k][1];
        float w = (float)watermark[base_wm + k];
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
            out_bgr[idx + 0] = (uchar)clampf(b, 0.0f, 255.0f);
            out_bgr[idx + 1] = (uchar)clampf(g, 0.0f, 255.0f);
            out_bgr[idx + 2] = (uchar)clampf(r, 0.0f, 255.0f);
        }
    }
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

__kernel void tpvm_apply(
    __global const uchar* in_bgr,
    __global uchar* out_a,
    __global uchar* out_b,
    int width,
    int height,
    float strength
) {
    int gid = get_global_id(0);
    int total = width * height;
    if (gid >= total) return;

    int x = gid % width;
    int y = gid / width;
    float p = ((x + y) & 1) ? 1.0f : -1.0f;
    float delta = p * strength;

    int idx = gid * 3;
    float b = (float)in_bgr[idx + 0];
    float g = (float)in_bgr[idx + 1];
    float r = (float)in_bgr[idx + 2];

    out_a[idx + 0] = (uchar)clampf(b + delta, 0.0f, 255.0f);
    out_a[idx + 1] = (uchar)clampf(g + delta, 0.0f, 255.0f);
    out_a[idx + 2] = (uchar)clampf(r + delta, 0.0f, 255.0f);

    out_b[idx + 0] = (uchar)clampf(b - delta, 0.0f, 255.0f);
    out_b[idx + 1] = (uchar)clampf(g - delta, 0.0f, 255.0f);
    out_b[idx + 2] = (uchar)clampf(r - delta, 0.0f, 255.0f);
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

int main() {
    const char* input_video = "sample3.mp4";
    const char* output_video = "hybrid_protected_output3.mp4";

    cv::VideoCapture cap(input_video);
    if (!cap.isOpened()) {
        printf("Input video not found: %s\n", input_video);
        printf("Place the video in E:\\OpenCL or update the path in src/security_gpu.cpp\n");
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    int width_trim = (width / 8) * 8;
    int height_trim = (height / 8) * 8;
    if (width_trim != width || height_trim != height) {
        printf("Warning: frame size not multiple of 8, cropping to %dx%d\n", width_trim, height_trim);
    }

    cv::VideoWriter out(
        output_video,
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps * 2.0,
        cv::Size(width_trim, height_trim)
    );
    if (!out.isOpened()) {
        printf("Failed to open output video: %s\n", output_video);
        return 1;
    }

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
        printf("No GPU device found. Exiting.\n");
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

    cl_kernel k_watermark = clCreateKernel(program, "watermark_block", &err);
    check_cl(err, "clCreateKernel(watermark_block)");
    cl_kernel k_detect = clCreateKernel(program, "detect_block", &err);
    check_cl(err, "clCreateKernel(detect_block)");
    cl_kernel k_tpvm = clCreateKernel(program, "tpvm_apply", &err);
    check_cl(err, "clCreateKernel(tpvm_apply)");

    int blocks_w = width_trim / 8;
    int blocks_h = height_trim / 8;
    int total_blocks = blocks_w * blocks_h;
    const int mid_len = 15;
    const float alpha = 10.0f;
    const float strength = 40.0f;

    const uint32_t watermark_key = 42;
    std::vector<char> watermark(total_blocks * mid_len);
    for (uint32_t i = 0; i < watermark.size(); ++i) {
        watermark[i] = wm_bit_from_index(i, watermark_key);
    }

    size_t frame_bytes = (size_t)width_trim * (size_t)height_trim * 3;
    cl_mem in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, frame_bytes, NULL, &err);
    check_cl(err, "clCreateBuffer(in_buf)");
    cl_mem wm_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, frame_bytes, NULL, &err);
    check_cl(err, "clCreateBuffer(wm_buf)");
    cl_mem out_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, frame_bytes, NULL, &err);
    check_cl(err, "clCreateBuffer(out_a)");
    cl_mem out_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, frame_bytes, NULL, &err);
    check_cl(err, "clCreateBuffer(out_b)");
    cl_mem wm_bits = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, watermark.size(), watermark.data(), &err);
    check_cl(err, "clCreateBuffer(wm_bits)");
    cl_mem block_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * total_blocks, NULL, &err);
    check_cl(err, "clCreateBuffer(block_sums)");

    std::vector<float> sums(total_blocks);

    cv::Mat frame;
    cv::Mat frame_trim(height_trim, width_trim, CV_8UC3);
    cv::Mat out_frame_a(height_trim, width_trim, CV_8UC3);
    cv::Mat out_frame_b(height_trim, width_trim, CV_8UC3);

    int frame_count = 0;
    printf("Starting GPU processing (Watermark + TPVM)...\n");

    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) break;

        if (frame.cols != width_trim || frame.rows != height_trim) {
            frame_trim = frame(cv::Rect(0, 0, width_trim, height_trim)).clone();
        } else {
            frame_trim = frame;
        }

        check_cl(clEnqueueWriteBuffer(queue, in_buf, CL_TRUE, 0, frame_bytes, frame_trim.data, 0, NULL, NULL), "clEnqueueWriteBuffer(in_buf)");

        check_cl(clSetKernelArg(k_watermark, 0, sizeof(cl_mem), &in_buf), "set k_watermark arg0");
        check_cl(clSetKernelArg(k_watermark, 1, sizeof(cl_mem), &wm_buf), "set k_watermark arg1");
        check_cl(clSetKernelArg(k_watermark, 2, sizeof(int), &width_trim), "set k_watermark arg2");
        check_cl(clSetKernelArg(k_watermark, 3, sizeof(int), &height_trim), "set k_watermark arg3");
        check_cl(clSetKernelArg(k_watermark, 4, sizeof(int), &blocks_w), "set k_watermark arg4");
        check_cl(clSetKernelArg(k_watermark, 5, sizeof(cl_mem), &wm_bits), "set k_watermark arg5");
        check_cl(clSetKernelArg(k_watermark, 6, sizeof(int), &mid_len), "set k_watermark arg6");
        check_cl(clSetKernelArg(k_watermark, 7, sizeof(float), &alpha), "set k_watermark arg7");

        size_t g_blocks = (size_t)total_blocks;
        check_cl(clEnqueueNDRangeKernel(queue, k_watermark, 1, NULL, &g_blocks, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(k_watermark)");

        check_cl(clSetKernelArg(k_tpvm, 0, sizeof(cl_mem), &wm_buf), "set k_tpvm arg0");
        check_cl(clSetKernelArg(k_tpvm, 1, sizeof(cl_mem), &out_a), "set k_tpvm arg1");
        check_cl(clSetKernelArg(k_tpvm, 2, sizeof(cl_mem), &out_b), "set k_tpvm arg2");
        check_cl(clSetKernelArg(k_tpvm, 3, sizeof(int), &width_trim), "set k_tpvm arg3");
        check_cl(clSetKernelArg(k_tpvm, 4, sizeof(int), &height_trim), "set k_tpvm arg4");
        check_cl(clSetKernelArg(k_tpvm, 5, sizeof(float), &strength), "set k_tpvm arg5");

        size_t g_pixels = (size_t)width_trim * (size_t)height_trim;
        check_cl(clEnqueueNDRangeKernel(queue, k_tpvm, 1, NULL, &g_pixels, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(k_tpvm)");

        if (frame_count % 10 == 0) {
            check_cl(clSetKernelArg(k_detect, 0, sizeof(cl_mem), &wm_buf), "set k_detect arg0");
            check_cl(clSetKernelArg(k_detect, 1, sizeof(int), &width_trim), "set k_detect arg1");
            check_cl(clSetKernelArg(k_detect, 2, sizeof(int), &height_trim), "set k_detect arg2");
            check_cl(clSetKernelArg(k_detect, 3, sizeof(int), &blocks_w), "set k_detect arg3");
            check_cl(clSetKernelArg(k_detect, 4, sizeof(cl_mem), &wm_bits), "set k_detect arg4");
            check_cl(clSetKernelArg(k_detect, 5, sizeof(int), &mid_len), "set k_detect arg5");
            check_cl(clSetKernelArg(k_detect, 6, sizeof(cl_mem), &block_sums), "set k_detect arg6");

            check_cl(clEnqueueNDRangeKernel(queue, k_detect, 1, NULL, &g_blocks, NULL, 0, NULL, NULL), "clEnqueueNDRangeKernel(k_detect)");
            check_cl(clEnqueueReadBuffer(queue, block_sums, CL_TRUE, 0, sizeof(float) * total_blocks, sums.data(), 0, NULL, NULL), "clEnqueueReadBuffer(block_sums)");

            double total = 0.0;
            for (int i = 0; i < total_blocks; ++i) total += sums[i];
            double corr = total / (double)(total_blocks * mid_len);
            printf("Frame %d: WM Correlation in Frame A = %.2f (Threshold > 2.0 usually)\n", frame_count, corr);
        }

        check_cl(clEnqueueReadBuffer(queue, out_a, CL_TRUE, 0, frame_bytes, out_frame_a.data, 0, NULL, NULL), "clEnqueueReadBuffer(out_a)");
        check_cl(clEnqueueReadBuffer(queue, out_b, CL_TRUE, 0, frame_bytes, out_frame_b.data, 0, NULL, NULL), "clEnqueueReadBuffer(out_b)");

        out.write(out_frame_a);
        out.write(out_frame_b);

        frame_count++;
    }

    clReleaseMemObject(in_buf);
    clReleaseMemObject(wm_buf);
    clReleaseMemObject(out_a);
    clReleaseMemObject(out_b);
    clReleaseMemObject(wm_bits);
    clReleaseMemObject(block_sums);
    clReleaseKernel(k_watermark);
    clReleaseKernel(k_detect);
    clReleaseKernel(k_tpvm);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    cap.release();
    out.release();

    printf("Processing complete. Saved to %s\n", output_video);
    return 0;
}
