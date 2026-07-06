#include "segment.h"


// 辅助函数: 计算 IoU
__device__ float calcIoU(const float x11, const float y11, const float x12, const float y12,
                         const float x21, const float y21, const float x22, const float y22) {
    const float xx1 = fmaxf(x11, x21);
    const float yy1 = fmaxf(y11, y21);
    const float xx2 = fminf(x12, x22);
    const float yy2 = fminf(y12, y22);
    const float area1 = (x12 - x11) * (y12 - y11);
    const float area2 = (x22 - x21) * (y22 - y21);
    const float inter = fmaxf(0.0f, xx2 - xx1) * fmaxf(0.0f, yy2 - yy1);

    return inter / (area1 + area2 - inter + 1e-6f);
}

/**
 * @brief 简易并行 NMS 标记核函数
 * @param d_proposals: [960, 38]
 * @param d_keep_index: [960] 输出标记数组: 1为保留, 0为抑制
 * @param iou_threshold: 交并比阈值
 * @param conf_threshold: 置信度阈值
 * @param M: 960
 */
__global__ void cudaNMS(const float *__restrict__ d_proposals, int32_t *__restrict__ d_keep_index,
                        const float iou_threshold, const float conf_threshold, const int32_t M) {
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) {
        return;
    }

    // 获取当前框信息
    const float score = d_proposals[idx * PROBES_W + 4];
    if (score < conf_threshold) {   // 置信度阈值
        d_keep_index[idx] = 0;
        return;
    }

    const float x1 = d_proposals[idx * PROBES_W + 0];
    const float y1 = d_proposals[idx * PROBES_W + 1];
    const float x2 = d_proposals[idx * PROBES_W + 2];
    const float y2 = d_proposals[idx * PROBES_W + 3];

    // 遍历所有其他框, 与所有更高置信度的框比较
    bool suppressed = false;
    for (int32_t i = 0; i < M; ++i) {
        if (i == idx) {
            continue;
        }
        // 只有当另一个框的分数更高时, 才可能抑制当前框.
        if (d_proposals[i * PROBES_W + 4] < score) {
            continue;
        }

        // 仅与置信度更高的框比较
        const float ox1 = d_proposals[i * PROBES_W + 0];
        const float oy1 = d_proposals[i * PROBES_W + 1];
        const float ox2 = d_proposals[i * PROBES_W + 2];
        const float oy2 = d_proposals[i * PROBES_W + 3];
        if (calcIoU(x1, y1, x2, y2, ox1, oy1, ox2, oy2) > iou_threshold) {
            suppressed = true;
            break;
        }
    }

    d_keep_index[idx] = suppressed ? 0 : 1;
}

/**
 * @brief 并行 NMS 标记核函数
 * @param d_proposals: [960, 38]
 * @param d_keep_index: 输出标记数组: 1为保留, 0为抑制
 * @param iou_threshold: 交并比阈值
 * @param conf_threshold: 置信度阈值
 * @param M: 960
 */
void launchNMS(const float *d_proposals, int32_t *d_keep_index, const float iou_threshold, const float conf_threshold, const int32_t M) {
    dim3 blockDim(256);
    dim3 gridSize((M + 255) / 256);
    cudaNMS<<<gridSize, blockDim>>>(d_proposals, d_keep_index, iou_threshold, conf_threshold, M);
}