#include "segment.h"


// 双线性插值辅助函数(Bilinear Upsampling)
__device__ float bilinear_interpolate(const float *masks, const int32_t H, const int32_t W, const float y, const float x) {
    // H = 304, W = 480
    // 1. 边界检查: 如果超出有效范围 [0, H-1] x [0, W-1], 返回 0
    // 注意: 允许访问到 H-1 和 W-1 的邻居, 所以限制是 < 0 或 > H-1
    if (x < 0 || x > static_cast<float>(W - 1) || y < 0 || y > static_cast<float>(H - 1)) {
        return 0.0f;
    }

    // 2. 获取左上角整数坐标
    const int32_t x0 = static_cast<int32_t>(floorf(x));
    const int32_t y0 = static_cast<int32_t>(floorf(y));

    // 3. 获取右下角整数坐标, 防止越界
    const int32_t x1 = min(x0 + 1, W - 1);
    const int32_t y1 = min(y0 + 1, H - 1);

    // 4. 计算小数部分权重
    const float dx = x - static_cast<float>(x0);
    const float dy = y - static_cast<float>(y0);

    // 5. 获取四个邻近点的值 (行优先存储: index = y × W + x)
    const float v_ll = masks[y0 * W + x0];
    const float v_lh = masks[y0 * W + x1];
    const float v_hl = masks[y1 * W + x0];
    const float v_hh = masks[y1 * W + x1];

    // 6. 双线性插值公式
    return v_ll * (1 - dy) * (1 - dx) +
           v_lh * (1 - dy) * (    dx) +
           v_hl * (    dy) * (1 - dx) +
           v_hh * (    dy) * (    dx);
}

// CUDA核函数: 对每个掩膜应用二值化阈值, 并与边界框对齐裁剪
__global__ void cudaCropAndUpscale(const float *proposals,          // (1, 960, 38)
                                   const float *proto_masks,        // (960, 304, 480)
                                   uint8_t *final_masks,            // (960, 304, 480)
                                   const int32_t final_steps,       // 304 × 480
                                   const int32_t ideal_width,       // 304 × 480
                                   const float conf_threshold,      // 0.25f
                                   const float mask_threshold,      // 0.35f
                                   const int32_t image_h, const int32_t image_w,    // 1200 × 1920
                                   const int32_t proto_h, const int32_t proto_w,    // 304 × 480
                                   const float scale_xy,            // 1.0
                                   const float sampling,            // 0.25f
                                   const int32_t M, const int32_t N) {   // M = 960, N = proto_h × proto_w
    const int32_t row = blockIdx.x;  // 行索引, 范围: [0, 960)
    if (row >= M) {
        return;
    }
    if (proposals[row * PROBES_W + 4] < conf_threshold) {
        return;
    }

    // 边界框限制到图像坐标系, 模型推理图像可能缩放/补边.
    // image_h_: 720   padded_y_: 448  model_h_: 1024  proto_h: 256
    // image_w_: 1280  padded_x_: 0    model_w_: 1024  proto_w: 256
    //                 scale_xy_: 0.8  mask_ratio: 4
    // image_h_ × scale_xy_ + padded_y_ = model_h_ ÷ 4 = proto_h_
    // image_w_ × scale_xy_ + padded_x_ = model_w_ ÷ 4 = proto_w_
    const float image_x1 = min(max(proposals[row * PROBES_W + 0] / scale_xy, 0.f), static_cast<float>(image_w));
    const float image_y1 = min(max(proposals[row * PROBES_W + 1] / scale_xy, 0.f), static_cast<float>(image_h));
    const float image_x2 = min(max(proposals[row * PROBES_W + 2] / scale_xy, 0.f), static_cast<float>(image_w));
    const float image_y2 = min(max(proposals[row * PROBES_W + 3] / scale_xy, 0.f), static_cast<float>(image_h));
    const float image_bw = image_x2 - image_x1;
    const float image_bh = image_y2 - image_y1;
    if (image_bh < 4.f || image_bw < 4.f) {
        return;
    }

    // 基于面积约束动态计算输出尺寸(四舍五入到最近整数)
    int32_t output_w = lrint(image_bw);
    int32_t output_h = lrint(image_bh);
    if (output_h * output_w > final_steps) {
        if (output_h > ideal_width && output_w > ideal_width) {
            // 情况1: 原矩形足够大, 裁剪理想正方形
            output_h = ideal_width;
            output_w = final_steps / output_h;
        } else if (output_h < ideal_width) {
            // 情况2: 高度受限 (H < ideal_width)
            output_w = final_steps / output_h;
        } else {
            // 情况3: 宽度受限 (W < ideal_width)
            output_h = final_steps / output_w;
        }
    }

    // 边界框缩放到掩码坐标系, 理论上肯定在有效范围内.
    const float proto_x1 = min(max(proposals[row * PROBES_W + 0] * sampling, 0.f), static_cast<float>(proto_w));
    const float proto_y1 = min(max(proposals[row * PROBES_W + 1] * sampling, 0.f), static_cast<float>(proto_h));
    const float proto_bw = output_w * scale_xy * sampling;
    const float proto_bh = output_h * scale_xy * sampling;

    // 当前线程的内存块
    const float *proto_mask = proto_masks + row * N;
    uint8_t *final_mask = final_masks + row * final_steps;

    // 多线程并行上采样
    // blockDim.x: 代表当前线程块的线程总数(256个线程).
    // threadIdx.x: 代表当前线程在线程块中的编号(0到255).
    const int32_t total_pixels = output_h * output_w;
    for (int32_t idx = threadIdx.x; idx < total_pixels; idx += blockDim.x) {
        const int32_t output_x = idx % output_w;
        const int32_t output_y = idx / output_w;

        // 使用实际输出尺寸 output_w/output_h 计算相对坐标, 保证采样点均匀分布在整个bbox区域内.
        const float rel_x = static_cast<float>(output_x) / static_cast<float>(output_w);
        const float rel_y = static_cast<float>(output_y) / static_cast<float>(output_h);

        // 映射回原型掩码空间的浮点坐标
        // 使用 (rel + 0.5/size) 或者直接从边缘开始映射, 通常实例分割采用对齐角落或中心
        // 这里采用标准的双线性映射: 从 x1_m 到 x2_m 均匀采样 out_w 个点
        const float src_x = proto_x1 + proto_bw * rel_x;
        const float src_y = proto_y1 + proto_bh * rel_y;

        // 双线性插值采样
        const float val = bilinear_interpolate(proto_mask, proto_h, proto_w, src_y, src_x);

        // 步长为 N, 确保数据存储在每块掩码的左上角
        final_mask[idx] = val > mask_threshold ? 255 : 0;
    }
}

// 核函数中对应变量取值:
// blockDim.x = 256: 线程块大小‌, 表示每个block中包含256个并行线程, 在循环中作为‌步长(stride)‌使用.
// blockIdx.x = [0, 960): proposal索引‌, 代表当前线程块正在处理第几个检测框. 由于 gridSize = 960, 每个block负责一个proposal的所有像素计算.
// threadIdx.x = [0, 256): 线程局部索引‌, 表示当前线程在其所属 block 内的编号. 在循环中作为‌起始偏移(offset)‌使用.

/**
 * @brief YOLO26实例分割高分辨率掩码采样启动函数
 * @param d_proposals [960, 38] 格式: bbox(4), score(1), class(1), coef(32)
 * @param d_proto_masks [960, H×W] 输出掩码
 * @param d_final_masks [960, final_steps] 输出掩码
 * @param final_steps  输出掩码步长, 最大边界框像素数
 * @param ideal_width  理想正方形的边长
 * @param conf_threshold 置信度阈值, 低于此值的框将被忽略
 * @param mask_threshold 二值化阈值, 低于此值的掩码被置零
 * @param image_h 原始图像高度 1200
 * @param image_w 原始图像宽度 1920
 * @param proto_h 掩码原型高度 304
 * @param proto_w 掩码原型宽度 480
 * @param scale_xy 模型输入缩放 1.0
 * @param sampling 原型掩码下采样比例 0.25
 * @param M 固定为 960, num of proposals
 * @param N 固定为 proto_h × proto_w = 145920
 */
void launchScale(const float *d_proposals, const float *d_proto_masks, uint8_t *d_final_masks,
                 const int32_t final_steps, const int32_t ideal_width, const float conf_threshold, const float mask_threshold,
                 const int32_t image_h, const int32_t image_w, const int32_t proto_h, const int32_t proto_w,
                 const float scale_xy, const float sampling, const int32_t M, const int32_t N) {
    // 边界框裁剪 + 上采样 + 二值化
    int32_t gridSize = M;       // 960个数据块
    int32_t blockDim = 512;     // 单块并行线程
    cudaCropAndUpscale<<<gridSize, blockDim>>>(d_proposals, d_proto_masks, d_final_masks, final_steps, ideal_width,
                                               conf_threshold, mask_threshold, image_h, image_w, proto_h, proto_w,
                                               scale_xy, sampling, M, N);

    cudaDeviceSynchronize();
}