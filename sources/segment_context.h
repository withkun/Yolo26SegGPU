#ifndef __INC_SEGMENT_CONTEXT_H
#define __INC_SEGMENT_CONTEXT_H

#include "nvinfer_utils.h"
#include "segment_utils.h"


#define INPUT_BLOB_NAME         "images"
#define OUTPUT1_BLOB_NAME       "output0"
#define OUTPUT2_BLOB_NAME       "output1"

#define IOU_THRESHOLD           0.75f           // 交并比阈值
#define CONF_THRESHOLD          0.55f           // 置信度阈值
#define MASK_THRESHOLD          0.45f           // 掩码二值化阈值

#define FINAL_STEPS             (3*320*480)     // 最大实例像素(应该可配置)


class SegmentContext {
public:
    SegmentContext() = default;
    SegmentContext(nvinfer1::ICudaEngine *engine, const nvinfer1::Dims &dims);
    ~SegmentContext();

    // 禁止拷贝
    SegmentContext(const SegmentContext &) = delete;
    SegmentContext &operator=(const SegmentContext &) = delete;

    // 移动构造与移动赋值
    SegmentContext(SegmentContext &&rsh) noexcept;
    SegmentContext &operator=(SegmentContext &&rsh) noexcept;

    void create_context(nvinfer1::ICudaEngine *engine, const nvinfer1::Dims &dims);
    void destroy_context();

    DetectResults RunSync(const cv::Mat &image);


protected:
    void letterbox(const cv::Mat &image);
    void inference();
    void postprocess(const cv::Mat &image, float iou_threshold, float conf_threshold, float mask_threshold);

    void trackers(float mask_threshold);

    static int32_t save(int32_t track_id, const std::vector<cv::Rect2d> &tracks);


private:
    cudaStream_t                stream_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    nvinfer1::Dims              input_dims_{};                  // nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}}
    nvinfer1::Dims              probe_dims_{};                  // nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}}
    nvinfer1::Dims              proto_dims_{};                  // nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}}

    float                      *d_image_{nullptr};
    float                      *d_proposals_{nullptr};
    float                      *d_prototypes_{nullptr};
    cv::Mat                     h_image_;
    std::vector<float>          h_proposals_;
    std::vector<float>          h_prototypes_;
    std::vector<DetectResult>   results_;

    // Input Tensor:            nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}}
    int32_t                     INPUT_N_{1};
    int32_t                     INPUT_C_{1};
    int32_t                     INPUT_H_{1024};
    int32_t                     INPUT_W_{1024};
    // Proposals Tensor:        nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}}
    int32_t                     PROBE_N_{1};
    int32_t                     PROBE_H_{960};
    int32_t                     PROBE_W_{38};
    // Prototypes Tensor:       nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}}
    int32_t                     PROTO_N_{1};
    int32_t                     PROTO_C_{32};
    int32_t                     PROTO_H_{256};
    int32_t                     PROTO_W_{256};

    int32_t                     INPUT_SIZE_{INPUT_N_ * INPUT_C_ * INPUT_H_ * INPUT_W_};
    int32_t                     PROBE_SIZE_{PROBE_N_ * PROBE_H_ * PROBE_W_};
    int32_t                     PROTO_SIZE_{PROTO_N_ * PROTO_C_ * PROTO_H_ * PROTO_W_};

    int32_t                     image_h_{0};
    int32_t                     image_w_{0};
    int32_t                     padded_x_{0};
    int32_t                     padded_y_{0};
    float                       scale_xy_{1.0f};

    // GPU解码相关
    float                       sampling_{0.25f};               // 原型掩码采样
    std::vector<int32_t>        h_keep_index_;                  // 有效索引标记
    int32_t                    *d_keep_index_{nullptr};         // 有效索引标记
    float                      *d_proto_masks_{nullptr};        // 低分辨率掩码(PROBES_N_ * PROTOS_H_ * PROTOS_W_)
    uint8_t                    *d_final_masks_{nullptr};        // 高分辨率掩码(PROBES_N_ * FINAL_STEPS)
    std::vector<uint8_t>        h_final_masks_;
    int32_t                     final_steps_{PROTO_H_ * PROTO_W_};
    int32_t                     ideal_width_{};
};
#endif //__INC_SEGMENT_CONTEXT_H