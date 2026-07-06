#include "segment.h"


// 通用矩阵向量乘法GEMV: General Matrix-Vector Multiplication

/**
 * @brief YOLO26低分辨率掩码生成核函数
 * @param proposals [960, 38] 包含bbox(4), score(1), class(1), coef(32)
 * @param prototypes [32, H×W] 原型基矩阵
 * @param d_keep_index [960] 标记数组: 1为保留, 0为抑制
 * @param proto_masks [960, H×W] 输出掩码
 * @param M 固定为 960, max detections / num of proposals
 * @param N 固定为 304 × 480 = 145920,
 */
__global__ void cudaGEMV(const float *__restrict__ proposals, const float *__restrict__ prototypes, const int32_t *__restrict__ d_keep_index,
                         float *__restrict__ proto_masks, const int32_t M, const int32_t N) {
    // 每个Block处理一个检测框
    const int32_t row = blockIdx.x;  // 行索引, 范围: [0, 960)
    if (row >= M) {
        return;
    }

    // 1. 置信度过滤
    // proposals: [x1,y1,x2,y2, score, class, coef_0,coef_1,...,coef_31]
    if (d_keep_index[row] != 1) {
        return;
    }

    // 2. 从proposals中提取当前框的32个系数到共享内存
    const int32_t tid = threadIdx.x;
    __shared__ float s_coefs[COEFFS_W];
    if (tid < COEFFS_W) {
        s_coefs[tid] = proposals[row * PROBES_W + 6 + tid];
    }
    __syncthreads();

    // 3. 并行计算掩码像素: Sum(coef[j] × proto[j][i])
    for (int32_t i = tid; i < N; i += blockDim.x) {
        float val = 0.0f;
        #pragma unroll
        for (int32_t k = 0; k < COEFFS_W; ++k) {
            val += s_coefs[k] * prototypes[k * N + i];
        }

        // 4. 写入全局内存
        proto_masks[row * N + i] = val;
    }
}

/**
 * @brief YOLO26低分辨率掩码生成启动函数
 * @param d_proposals [960, 38] 格式: bbox(4), score(1), class(1), coef(32)
 * @param d_prototypes [32, H×W] 原型基矩阵
 * @param d_keep_index [960] 标记数组: 1为保留, 0为抑制
 * @param d_proto_masks [960, H×W] 输出掩码
 * @param proto_h 掩码原型高度 304
 * @param proto_w 掩码原型宽度 480
 * @param M 固定为 960, num of proposals
 * @param P 固定为 38, len of proposal
 * @param K 固定为 32, len of coefficient
 * @param N 固定为 H * W = 145920,
 */
void launchGEMV(const float *d_proposals, const float *d_prototypes, const int32_t *d_keep_index, float *d_proto_masks,
                const int32_t proto_h, const int32_t proto_w, const int32_t M, const int32_t P, const int32_t K, const int32_t N) {
    dim3 blockDim(512, 1, 1);     // 单块并行线程
    dim3 gridSize(M, 1, 1);       // 960个数据块
    cudaGEMV<<<gridSize, blockDim, 0>>>(d_proposals, d_prototypes, d_keep_index, d_proto_masks, M, N);

    // 同步并获取结果
    cudaDeviceSynchronize();
}