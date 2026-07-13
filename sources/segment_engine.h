#ifndef __INC_SEGMENT_ENGINE_H
#define __INC_SEGMENT_ENGINE_H

#include "segment_context.h"


class SegmentEngine {
public:
    explicit SegmentEngine();
    ~SegmentEngine();

    // 禁止拷贝
    SegmentEngine(const SegmentEngine &) = delete;
    SegmentEngine &operator=(const SegmentEngine &) = delete;

    bool                        get_engine(const std::string &model_file, const std::map<std::string, std::vector<nvinfer1::Dims>> &dimensions);
    void                        create_context();

    SegmentResults              RunSync(const cv::Mat &image);

protected:
    bool load_network_onnx(const std::string &model_file, const std::map<std::string, std::vector<nvinfer1::Dims>> &dimensions);
    bool load_network_engine(const std::string &engine_file);
    void get_model_dimensions();

private:
    std::unique_ptr<nvinfer1::IRuntime>     runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine>  engine_{nullptr};
    std::vector<SegmentContext>             contexts_;

    const int32_t               input_index_{0};
    const int32_t               probe_index_{1};
    const int32_t               proto_index_{2};
};
#endif //__INC_SEGMENT_ENGINE_H