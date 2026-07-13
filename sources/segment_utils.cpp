#include "segment_utils.h"


std::vector<cv::Scalar> label_colormap() {
    std::vector<cv::Scalar> colormap(256);
    for (int i = 0; i < 256; ++i) {
        // 提取标签i的8个二进制位
        const uint8_t b0 = (i >> 0) & 1;
        const uint8_t b1 = (i >> 1) & 1;
        const uint8_t b2 = (i >> 2) & 1;
        const uint8_t b3 = (i >> 3) & 1;
        const uint8_t b4 = (i >> 4) & 1;
        const uint8_t b5 = (i >> 5) & 1;
        const uint8_t b6 = (i >> 6) & 1;
        const uint8_t b7 = (i >> 7) & 1;
        // 合成RGB通道色彩值.
        const uint8_t r = (b0 << 7) | (b3 << 6) | (b6 << 5);
        const uint8_t g = (b1 << 7) | (b4 << 6) | (b7 << 5);
        const uint8_t b = (b2 << 7) | (b5 << 6);
        colormap[i] = cv::Scalar(b, g, r);
    }
    return colormap;
}

void DrawPred(cv::Mat &image, const std::vector<SegmentResult> &results, int32_t index) {
    const float fontScale = 0.5f;
    const int32_t thickness = 1;
    const int32_t fontFace = cv::FONT_HERSHEY_COMPLEX;
    const int32_t lineType = cv::LINE_AA;
    const cv::Vec3i red(0, 0, 255);

    // 转三通道绘图
    const static std::vector<cv::Scalar> LABEL_COLORMAP = label_colormap();
    if (image.type() == CV_8UC1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }

    const cv::Mat all_mask = image.clone();
    for (const auto &&[idx, result] : results | std::views::enumerate) {
        const auto &[score, label, bbox, mask, id] = result;
        const auto &color = LABEL_COLORMAP[idx % LABEL_COLORMAP.size()];
        cv::rectangle(image, bbox, color, thickness, lineType);

        // mask是一个与矩阵box大小相同的单通道二值掩码
        if (!mask.empty()) {
            all_mask(bbox).setTo(color, mask);
        }

        const cv::Point center(bbox.x + 0.5 * bbox.width, bbox.y + 0.5 * bbox.height);

        int32_t baseLine;
        const std::string text = std::format("{}:{:.3f} {}:{}", label, score, center.x, center.y);
        const cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
        const cv::Point point(bbox.x, std::max(bbox.y, textSize.height));
        cv::putText(image, text, point, fontFace, fontScale, color, thickness, lineType);
        if (id > 0) {
            const std::string info = std::format("{}", id);
            cv::putText(image, info, cv::Point(bbox.x, bbox.y+textSize.height), fontFace, fontScale, red, thickness, lineType);
        }
    }

    cv::addWeighted(image, 0.7, all_mask, 0.3, 0, image); //将mask加在原图上面
}