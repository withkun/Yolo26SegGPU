#ifndef __INC_SEGMENT_ENGINE_H
#define __INC_SEGMENT_ENGINE_H

#include "segment_context.h"

#include <future>


// 线程安全的上下文管理示例
struct Task {
    cv::Mat                         image_;
    std::promise<DetectResults>     results_;
};

// Instance Segmentation
class SegmentEngine {
public:
    explicit SegmentEngine();
    ~SegmentEngine();

    // 禁止拷贝
    SegmentEngine(const SegmentEngine &) = delete;
    SegmentEngine &operator=(const SegmentEngine &) = delete;

    bool                        get_engine(const std::string &model_file, const std::map<std::string, std::vector<nvinfer1::Dims>> &dimensions);
    void                        create_context(const nvinfer1::Dims &dims);

    DetectResults               RunSync(const nvinfer1::Dims &kDims, const cv::Mat &image);

private:
    bool load_network_onnx(const std::string &model_file, const std::map<std::string, std::vector<nvinfer1::Dims>> &dimensions);
    bool load_network_engine(const std::string &engine_file);
    void get_model_dimensions();

protected:
    nvinfer1::IRuntime         *runtime_{nullptr};
    nvinfer1::ICudaEngine      *engine_{nullptr};
    std::map<nvinfer1::Dims, SegmentContext, DimsCompare>    contexts_;

    const int32_t               input_index_{0};
    const int32_t               probe_index_{1};
    const int32_t               proto_index_{2};
};
#endif  //__INC_SEGMENT_ENGINE_H