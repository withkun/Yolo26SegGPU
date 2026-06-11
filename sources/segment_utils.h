#ifndef __INC_SEGMENT_UTILS_H
#define __INC_SEGMENT_UTILS_H

#include "opencv2/opencv.hpp"
#include <ranges>


#define P_SWAP(A, B) do { \
    auto *T = A; A = B; B = T; \
} while (false)


// 边界框格式:
//  xywh: 其中前两个xy是中心点坐标, w是宽度, h是高度.
//  xyxy: 其中前两个xy是左上角坐标, 后两个xy是右下角坐标.
//  xyah: 其中前两个xy是中心点坐标, a是宽高比, h是高度.
//  xysr: 其中前两个xy是中心点坐标, s是面积尺度, r是宽高比.
struct DetectResult {
    float           score;      //结果置信度
    int32_t         label;      //结果类型Id
    cv::Rect2i      bbox;       //目标边界框
    cv::Mat         mask;       //目标掩码

    int32_t         id;         //跟踪标识
};
using DetectResults = std::vector<DetectResult>;


void DrawPred(cv::Mat &image, const std::vector<DetectResult> &results, int32_t index = 0);


inline std::string trim(const std::string &s) {
    if (s.empty()) {
        return s;
    }

    // 常见空白字符(C风格isspace范围)
    static constexpr std::string WHITESPACE = " \t\n\r\f\v";
    const size_t s_idx = s.find_first_not_of(WHITESPACE);
    if (s_idx == std::string::npos) {    // 全空白
        return "";
    }

    const size_t e_idx = s.find_last_not_of(WHITESPACE);
    return s.substr(s_idx, (e_idx - s_idx) + 1);
}
#endif //__INC_SEGMENT_UTILS_H