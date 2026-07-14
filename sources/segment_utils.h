#ifndef __INC_SEGMENT_UTILS_H
#define __INC_SEGMENT_UTILS_H

#include "opencv2/opencv.hpp"
#include <ranges>


#define ROUND_UP(N, X)          int32_t((N + X - 1) / X) * X

#define P_SWAP(A, B) do { \
    auto *T = A; A = B; B = T; \
} while (false)


// 边界框格式:
//  xywh: 其中前两个xy是中心点坐标, w是宽度, h是高度.
//  xyxy: 其中前两个xy是左上角坐标, 后两个xy是右下角坐标.
//  xyah: 其中前两个xy是中心点坐标, a是宽高比, h是高度.
//  xysr: 其中前两个xy是中心点坐标, s是面积尺度, r是宽高比.
struct SegmentResult {
    float                       score;      //结果置信度
    int32_t                     label;      //结果类型Id
    cv::Rect2i                  bbox;       //目标边界框
    cv::Mat                     mask;       //目标掩码

    int32_t                     id;         //跟踪标识
};
using SegmentResults = std::vector<SegmentResult>;


void DrawPred(cv::Mat &image, const std::vector<SegmentResult> &results, int32_t index = 0);


#endif //__INC_SEGMENT_UTILS_H