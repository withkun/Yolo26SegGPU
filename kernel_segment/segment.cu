#include "segment.h"

#include <chrono>
#include <iostream>


/**
 * @brief YOLO26实例分割后处理函数
 * @param d_proposals [960, 38] 格式: bbox(4), score(1), class(1), coef(32)
 * @param d_prototypes [32, H×W] 原型基矩阵
 * @param d_proto_masks [960, H×W] 输出掩码
 * @param d_final_masks [960, final_steps] 输出掩码
 * @param h_final_masks [960, final_steps] 输出掩码
 * @param final_steps  输出掩码步长, 最大边界框像素数
 * @param ideal_width  输出掩码步长, 最大边界框像素数
 * @param conf_threshold 置信度阈值, 低于此值的框将被忽略
 * @param mask_threshold 二值化阈值, 低于此值的掩码被置零
 * @param image_h 原始图像高度 1200
 * @param image_w 原始图像宽度 1920
 * @param proto_h 模型输出掩码高度(PROTOS_H_): 304
 * @param proto_w 模型输出掩码宽度(PROTOS_W_): 480
 * @param scale_xy 模型输入缩放 1.0
 * @param sampling 原型掩码下采样比例 0.25
 * @param M 固定为(PROBES_H_): 960, num of proposals
 * @param P 固定为(PROBES_W_): 38, len of proposal
 * @param K 固定为32, len of coefficient
 */
void cudaPostprocess(const float *d_proposals, const float *d_prototypes, float *d_proto_masks, uint8_t *d_final_masks, uint8_t *h_final_masks, const int32_t final_steps, const int32_t ideal_width,
                     const float conf_threshold, const float mask_threshold, const int32_t image_h, const int32_t image_w, const int32_t proto_h, const int32_t proto_w,
                     const float scale_xy, const float sampling, const int32_t M, const int32_t P, const int32_t K,
                     int64_t &usage1, int64_t &usage2, int64_t &usage3) {
    // 0. 参数有效性判断
    if (P != PROBES_W || K != COEFFS_W) {
        std::cerr << "P and K must be the same" << std::endl;
        throw std::invalid_argument("P and K must be the same");
    }
    const int32_t N = proto_h * proto_w;    // 掩码原型步长, 固定为 H × W = 145920

    // 1. 重建低分辨率掩膜
    const auto t1 = std::chrono::system_clock::now();
    launchGEMV(d_proposals, d_prototypes, d_proto_masks, conf_threshold, proto_h, proto_w, M, P, K, N);

    // 2. 重建高分辨率掩膜
    const auto t2 = std::chrono::system_clock::now();
    launchScale(d_proposals, d_proto_masks, d_final_masks, final_steps, ideal_width, conf_threshold, mask_threshold, image_h, image_w, proto_h, proto_w, scale_xy, sampling, M, N);

    // 3. 拷贝输出数据到主机
    const auto t3 = std::chrono::system_clock::now();
    cudaMemcpy(h_final_masks, d_final_masks, M * final_steps * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 4. 耗时统计日志输出
    const auto t4 = std::chrono::system_clock::now();
    //std::cerr << std::format("launchGEMV usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) << std::endl;
    //std::cerr << std::format("launchCrop usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count()) << std::endl;
    //std::cerr << std::format("launchCrop usage: {}μs", std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) << std::endl;
    usage1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    usage2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    usage3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
}