#include "segment_context.h"
#include "kernel_segment/segment.h"

#include <fstream>
#include <filesystem>


SegmentContext::SegmentContext(nvinfer1::ICudaEngine *engine) {
    create_context(engine);
}

SegmentContext::~SegmentContext() {
    destroy_context();
}

SegmentContext::SegmentContext(SegmentContext &&rsh) noexcept {
    *this = std::move(rsh);
    rsh.stream_ = nullptr;
}

SegmentContext &SegmentContext::operator=(SegmentContext &&rsh) noexcept {
    this->stream_               = rsh.stream_;
    this->context_              = std::move(rsh.context_);

    this->d_image_              = rsh.d_image_;
    this->d_proposals_          = rsh.d_proposals_;
    this->d_prototypes_         = rsh.d_prototypes_;
    this->h_image_              = rsh.h_image_;
    this->h_proposals_          = rsh.h_proposals_;
    this->h_prototypes_         = rsh.h_prototypes_;
    this->results_              = rsh.results_;

    // Input Tensor:            nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}}
    this->MAX_IMAGE_N_          = rsh.MAX_IMAGE_N_;
    this->MAX_IMAGE_C_          = rsh.MAX_IMAGE_C_;
    this->MAX_IMAGE_H_          = rsh.MAX_IMAGE_H_;
    this->MAX_IMAGE_W_          = rsh.MAX_IMAGE_W_;
    // Proposals Tensor:        nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}}
    this->MAX_PROPO_N_          = rsh.MAX_PROPO_N_;
    this->MAX_PROPO_H_          = rsh.MAX_PROPO_H_;
    this->MAX_PROPO_W_          = rsh.MAX_PROPO_W_;
    // Prototypes Tensor:       nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}}
    this->MAX_PROTO_N_          = rsh.MAX_PROTO_N_;
    this->MAX_PROTO_C_          = rsh.MAX_PROTO_C_;
    this->MAX_PROTO_H_          = rsh.MAX_PROTO_H_;
    this->MAX_PROTO_W_          = rsh.MAX_PROTO_W_;

    this->MAX_IMAGE_SIZE_       = rsh.MAX_IMAGE_SIZE_;
    this->MAX_PROPO_SIZE_       = rsh.MAX_PROPO_SIZE_;
    this->MAX_PROTO_SIZE_       = rsh.MAX_PROTO_SIZE_;

    this->image_h_              = rsh.image_h_;
    this->image_w_              = rsh.image_w_;
    this->padded_x_             = rsh.padded_x_;
    this->padded_y_             = rsh.padded_y_;
    this->scale_xy_             = rsh.scale_xy_;

    // GPU解码相关
    this->sampling_             = rsh.sampling_;
    this->h_keep_index_         = rsh.h_keep_index_;
    this->d_keep_index_         = rsh.d_keep_index_;
    this->d_proto_masks_        = rsh.d_proto_masks_;
    this->d_final_masks_        = rsh.d_final_masks_;
    this->h_final_masks_        = rsh.h_final_masks_;
    this->final_steps_          = rsh.final_steps_;
    this->ideal_width_          = rsh.ideal_width_;

    rsh.stream_ = nullptr;
    return *this;
}

void SegmentContext::create_context(nvinfer1::ICudaEngine *engine) {
    const auto time_point1 = std::chrono::system_clock::now();
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    // Dynamic Shape 模型肯定是绑定过Profile的, 根据最大尺寸分配空间.
    if (engine->getNbOptimizationProfiles() > 0) {
        const auto kDims = engine->getProfileShape(INPUT_BLOB_NAME, 0, nvinfer1::OptProfileSelector::kMAX);
        context_->setInputShape(INPUT_BLOB_NAME, kDims);
    }

    // 计算输入输出尺寸.
    std::vector<const char *> tensor_names{ INPUT_BLOB_NAME, OUTPUT1_BLOB_NAME, OUTPUT2_BLOB_NAME };
    context_->inferShapes(tensor_names.size(), tensor_names.data());

    // 获取模型输入尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}})
    const auto image_dims = context_->getTensorShape(INPUT_BLOB_NAME);
    MAX_IMAGE_N_ = static_cast<int32_t>(image_dims.d[0]);      // 1
    MAX_IMAGE_C_ = static_cast<int32_t>(image_dims.d[1]);      // 1
    MAX_IMAGE_H_ = static_cast<int32_t>(image_dims.d[2]);      // 1024
    MAX_IMAGE_W_ = static_cast<int32_t>(image_dims.d[3]);      // 1024
    MAX_IMAGE_SIZE_ = DimsInBytes(image_dims);
    SPDLOG_INFO("TensorRT image for OptProfileSelector::kMAX: {} count: {}", image_dims, MAX_IMAGE_SIZE_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}})
    const auto propo_dims = context_->getTensorShape(OUTPUT1_BLOB_NAME);
    MAX_PROPO_N_ = static_cast<int32_t>(propo_dims.d[0]);      // 1
    MAX_PROPO_H_ = static_cast<int32_t>(propo_dims.d[1]);      // 960
    MAX_PROPO_W_ = static_cast<int32_t>(propo_dims.d[2]);      // 38
    MAX_PROPO_SIZE_ = DimsInBytes(propo_dims);
    SPDLOG_INFO("TensorRT output1 for OptProfileSelector::kMAX: {} count: {}", propo_dims, MAX_PROPO_SIZE_);

    // 获取输出尺寸并分配GPU内存 (nvinfer1::Dims{nbDims=4, d={1, 32, 256, 256, 0, 0, 0, 0}})
    const auto proto_dims = context_->getTensorShape(OUTPUT2_BLOB_NAME);
    MAX_PROTO_N_ = static_cast<int32_t>(proto_dims.d[0]);      // 1
    MAX_PROTO_C_ = static_cast<int32_t>(proto_dims.d[1]);      // 32
    MAX_PROTO_H_ = static_cast<int32_t>(proto_dims.d[2]);      // 256
    MAX_PROTO_W_ = static_cast<int32_t>(proto_dims.d[3]);      // 256
    MAX_PROTO_SIZE_ = DimsInBytes(proto_dims);
    SPDLOG_INFO("TensorRT output2 for OptProfileSelector::kMAX: {} count: {}", proto_dims, MAX_PROTO_SIZE_);

    // 分配GPU内存, 绑定数据缓冲区.
    cudaStreamCreate(&stream_);
    cudaMalloc(&d_image_,      MAX_IMAGE_SIZE_ * sizeof(float));
    cudaMalloc(&d_proposals_,  MAX_PROPO_SIZE_ * sizeof(float));
    cudaMalloc(&d_prototypes_, MAX_PROTO_SIZE_ * sizeof(float));
    context_->setTensorAddress(INPUT_BLOB_NAME,   d_image_);
    context_->setTensorAddress(OUTPUT1_BLOB_NAME, d_proposals_);
    context_->setTensorAddress(OUTPUT2_BLOB_NAME, d_prototypes_);
    h_proposals_.resize(MAX_PROPO_SIZE_);
    h_prototypes_.resize(MAX_PROTO_SIZE_);
    SPDLOG_INFO("TensorRT Init, Context: {}:{} {}:{}:{}", static_cast<void *>(this), static_cast<void *>(context_.get()), static_cast<void *>(d_image_), static_cast<void *>(d_proposals_), static_cast<void *>(d_prototypes_));

    // GPU解码相关内存
    sampling_ = static_cast<float>(MAX_PROTO_W_) / static_cast<float>(MAX_IMAGE_W_);    // 下采样比例: 0.25
    h_keep_index_.resize(MAX_PROPO_H_);                                                 // [960] 有效索引标记
    cudaMalloc(&d_keep_index_, MAX_PROPO_H_ * sizeof(int32_t));                         // [960] 有效索引标记
    cudaMalloc(&d_proto_masks_, MAX_PROPO_H_ * MAX_PROTO_H_ * MAX_PROTO_W_ * sizeof(float));    // [960, 304*480]    534M 低分辨率掩膜, 需要单独分配
    if (d_proto_masks_ == nullptr) {
        std::cerr << "cudaMalloc d_proto_masks is null" << std::endl;
        throw std::runtime_error("cudaMalloc d_proto_masks is null");
    }

    final_steps_ = FINAL_STEPS;
    ideal_width_ = std::sqrt(final_steps_);
    h_final_masks_.resize(MAX_PROPO_H_ * final_steps_);
    cudaMalloc(&d_final_masks_, MAX_PROPO_H_ * final_steps_ * sizeof(uint8_t));         // [960, 304*480]    134M 高分辨率掩膜, 可以单独分配
    if (d_final_masks_ == nullptr) {
        std::cerr << "cudaMalloc d_final_masks is null" << std::endl;
        throw std::runtime_error("cudaMalloc d_final_masks is null");
    }

    const auto time_point2 = std::chrono::system_clock::now();
    SPDLOG_INFO("===> TensorRT Prepare buffer success, usage: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(time_point2 - time_point1).count());
}

void SegmentContext::destroy_context() {
    if (stream_ == nullptr) {
        return;
    }

    context_->setTensorAddress(INPUT_BLOB_NAME,   nullptr);
    context_->setTensorAddress(OUTPUT1_BLOB_NAME, nullptr);
    context_->setTensorAddress(OUTPUT2_BLOB_NAME, nullptr);
    cudaFree(d_image_);
    cudaFree(d_proposals_);
    cudaFree(d_prototypes_);

    // GPU解码相关内存
    cudaFree(d_keep_index_);
    cudaFree(d_proto_masks_);
    cudaFree(d_final_masks_);

    context_.reset();
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
}

void SegmentContext::letterbox(const cv::Mat &image, int32_t &round_h, int32_t &round_w) {
    image_h_ = image.rows;
    image_w_ = image.cols;

    cv::Mat final;
    if (image_h_ > MAX_IMAGE_H_ || image_w_ > MAX_IMAGE_W_) {
        // 图像预处理方法: 计算缩放比例, 取其中较小的一侧以保持原图的宽高比.
        round_h = MAX_IMAGE_H_;
        round_w = MAX_IMAGE_W_;
        scale_xy_ = std::min(1.0f * round_h / image_h_, 1.0f * round_w / image_w_);
        const auto scaled_h = static_cast<int32_t>(image_h_ * scale_xy_);
        const auto scaled_w = static_cast<int32_t>(image_w_ * scale_xy_);
        padded_y_ = round_h - scaled_h;
        padded_x_ = round_w - scaled_w;
        // 需要缩放, 需要填充
        cv::resize(image, final, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(final, final, 0, padded_y_, 0, padded_x_, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    } else if (image_h_ % 32 != 0 || image_w_ % 32 != 0) {
        // 是否需要对齐到32的倍数
        round_h = ROUND_UP(image_h_, 32);
        round_w = ROUND_UP(image_w_, 32);
        scale_xy_ = std::min(1.0f * round_h / image_h_, 1.0f * round_w / image_w_);
        const auto scaled_h = static_cast<int32_t>(image_h_ * scale_xy_);
        const auto scaled_w = static_cast<int32_t>(image_w_ * scale_xy_);
        padded_y_ = round_h - scaled_h;
        padded_x_ = round_w - scaled_w;
        cv::resize(image, final, cv::Size(scaled_w, scaled_h), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(final, final, 0, padded_y_, 0, padded_x_, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    } else {
        // 不要缩放, 不要填充
        round_h = image_h_;
        round_w = image_w_;
        scale_xy_ = 1.0f;
        padded_y_ = 0;
        padded_x_ = 0;
        final = image;
    }
    SPDLOG_INFO("TensorRT letterbox, image size: {} final size: {} scale_xy: {} padded_x: {} padded_y: {}", image.size, final.size, scale_xy_, padded_x_, padded_y_);

    // image: 输入图像, 灰度图或三通道图(一般为BGR).
    // blob: 输出4维矩阵, 符合模型输入的NCHW格式. [1, C, H, W]
    // scalefactor: 缩放因子, 图像像素值的缩放比例; 图像像素减去平均值之后, 再进行缩放, 默认值是1.
    // size: 目标尺寸, 模型输入的图片尺寸.
    // mean: 图像要减去均值, 如果需要对BGR图片的三个通道分别减去不同的值, 可以使用3个值; 如果三通道图像只有1个值, 那么三个通道都减去相同的值.
    // swapRB: OpenCV中图片通道顺序是BGR, 但是假设输入顺序是RGB, 处理时可以同步转换为RGB格式, 那么就要使swapRB=true.
    // crop: 是否裁剪, 调整尺寸时是保持比例并裁剪(非拉伸), 如果crop裁剪为true, 则调整输入图像的大小, 使调整大小后的一侧等于相应的尺寸, 另一侧等于或大于, 然后从中心进行裁剪; 如果crop裁剪为false, 则直接调整大小而不进行裁剪并保留纵横比.
    // ddepth: 输出数据类型, 通常为CV_32F或CV_8U.
    cv::dnn::blobFromImage(final, h_image_, 1.0/255.0, cv::Size(round_w, round_h), cv::Scalar(), true, false, CV_32F);
    SPDLOG_INFO("TensorRT letterbox final image size: {} total: {}", final.size, h_image_.total());
}

SegmentResults SegmentContext::RunSync(const cv::Mat &image) {
    int32_t scaled_h, scaled_w;
    letterbox(image, scaled_h, scaled_w);

    const nvinfer1::Dims kDims{ .nbDims = 4, .d = {1, 1, scaled_h, scaled_w, 0, 0, 0, 0}};  // (N, C, H, W)
    context_->setOptimizationProfileAsync(0, stream_);
    context_->setInputShape(INPUT_BLOB_NAME,  kDims);
    if (!context_->allInputDimensionsSpecified()) {
        throw std::runtime_error("Not all input dimensions specified");
    }

    inference();

    postprocess(image, IOU_THRESHOLD, CONF_THRESHOLD, MASK_THRESHOLD);

    return results_;
}

void SegmentContext::inference() {
    SPDLOG_INFO("TensorRT inference enter: {}", h_image_.total());
    // 异步流拷贝输入数据
    cudaMemcpyAsync(d_image_, h_image_.ptr<float>(0), h_image_.total() * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // 异步流提交推理任务
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("CUDA enqueueV3 not success");
    }

    // 异步流拷贝输出数据
    cudaMemcpyAsync(h_proposals_.data(), d_proposals_, MAX_PROPO_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_prototypes_.data(), d_prototypes_, MAX_PROTO_SIZE_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    // 流同步等待处理完成
    cudaStreamSynchronize(stream_);
    SPDLOG_INFO("TensorRT inference leave");
}

void SegmentContext::postprocess(const cv::Mat &image, const float iou_threshold, const float conf_threshold, const float mask_threshold) {
    // 推理已完成‌: 必须在 context->enqueueV3(stream) ‌之后‌, 且同步完成 (cudaStreamSynchronize)后查询, 否则形状可能未更新或不准确.
    const auto image_dims = context_->getTensorShape(INPUT_BLOB_NAME);      // nvinfer1::Dims{nbDims=4, d={1, 1, 1024, 1024, 0, 0, 0, 0}}
    const auto probe_dims = context_->getTensorShape(OUTPUT1_BLOB_NAME);    // nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}}
    const auto proto_dims = context_->getTensorShape(OUTPUT2_BLOB_NAME);    // nvinfer1::Dims{nbDims=4, d={1, 32, 200, 384, 0, 0, 0, 0}}
    SPDLOG_INFO("TensorRT image: {} probe: {}, proto: {}", image_dims, probe_dims, proto_dims);
    IMAGE_N_ = static_cast<int32_t>(image_dims.d[0]);         // 1
    IMAGE_C_ = static_cast<int32_t>(image_dims.d[1]);         // 1
    IMAGE_H_ = static_cast<int32_t>(image_dims.d[2]);         // 800
    IMAGE_W_ = static_cast<int32_t>(image_dims.d[3]);         // 1536
    PROPO_N_ = static_cast<int32_t>(probe_dims.d[0]);         // 1
    PROPO_H_ = static_cast<int32_t>(probe_dims.d[1]);         // 960
    PROPO_W_ = static_cast<int32_t>(probe_dims.d[2]);         // 38
    PROTO_N_ = static_cast<int32_t>(proto_dims.d[0]);         // 1
    PROTO_C_ = static_cast<int32_t>(proto_dims.d[1]);         // 32
    PROTO_H_ = static_cast<int32_t>(proto_dims.d[2]);         // 200
    PROTO_W_ = static_cast<int32_t>(proto_dims.d[3]);         // 384
    sampling_ = static_cast<float>(PROTO_W_) / static_cast<float>(IMAGE_W_);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // YOLO26模型输出:
    // output0输出为预测结果张量, 其维度是(1,960,38)  NHW: (x1,y1,x2,y2, score, class_id, mask1,mask2,...,mask32)
    // 38为4+1+1+32, 4为box的[x1,y1,x2,y2], 1个类别置信度, 1个类别标识, 32个掩膜系数(mask coefficients), 960个预测候选结果.
    // output1输出为掩膜原型张量, 其维度是(1,32,320,480), 32对应掩膜系数. 掩膜原型(320×480)需先上采样至输入图像尺寸(1280×1920).
    // 模型在推理时, 会根据每个检测框的32个掩膜系数, 对这些掩膜原型进行加权求和, 从而生成每个目标的掩膜.
    // YOLO26预测结果数据结构示例:
    // (nvinfer1::Dims{nbDims=3, d={1, 960, 38, 0, 0, 0, 0, 0}})   // 960行, 38列
    // 38为4+1+1+32, 4个边界框, 1个类别置信度, 1个类别标识, 32是掩膜系数
    // box1{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box2{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box3{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box4{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box5{ {x1,y1,x2,y2}, conf, class, {coef_0,coef_1,...,coef_31} }
    // box960
    SPDLOG_INFO(std::format("TensorRT offset_x:{} offset_y:{} scale_xy:{}", padded_x_, padded_y_, scale_xy_));

    // 注意: 送入模型的推理图像没有缩放, 只有补边, 如果有缩放下面实现需要调整.
    int64_t usage1, usage2, usage3;
    cudaPostprocess(d_proposals_, d_prototypes_, d_keep_index_, h_keep_index_.data(), d_proto_masks_, d_final_masks_, h_final_masks_.data(), final_steps_, ideal_width_,
                    iou_threshold, conf_threshold, mask_threshold, image_h_, image_w_, PROTO_H_, PROTO_W_, scale_xy_, sampling_, PROPO_H_, PROPO_W_, COEFFS_W,
                    usage1, usage2, usage3);
    SPDLOG_INFO(std::format("TensorRT probe usage: gemv={}μs scale:{}μs copy:{}μs", usage1, usage2, usage3));

    results_.clear();
    results_.reserve(PROPO_H_);

    // 根据置信度过滤(Proposals Tensor)
    const float *proposals = h_proposals_.data();
    for (int32_t row = 0; row < PROPO_H_; ++row) {
        if (h_keep_index_[row] != 1) {
            continue;
        }

        // 边界框限制到图像坐标系, 模型推理图像可能缩放/补边.
        // image_h_: 720   padded_y_: 448  model_h_: 1024  proto_h: 256
        // image_w_: 1280  padded_x_: 0    model_w_: 1024  proto_w: 256
        //                 scale_xy_: 0.8  mask_ratio: 4
        // image_h_ × scale_xy_ + padded_y_ = model_h_ ÷ 4 = proto_h_
        // image_w_ × scale_xy_ + padded_x_ = model_w_ ÷ 4 = proto_w_
        const float image_x1 = std::min(std::max(proposals[row * PROPO_W_ + 0] / scale_xy_, 0.f), static_cast<float>(image_w_));
        const float image_y1 = std::min(std::max(proposals[row * PROPO_W_ + 1] / scale_xy_, 0.f), static_cast<float>(image_h_));
        const float image_x2 = std::min(std::max(proposals[row * PROPO_W_ + 2] / scale_xy_, 0.f), static_cast<float>(image_w_));
        const float image_y2 = std::min(std::max(proposals[row * PROPO_W_ + 3] / scale_xy_, 0.f), static_cast<float>(image_h_));
        const float image_bw = image_x2 - image_x1;
        const float image_bh = image_y2 - image_y1;
        if (image_bh < 4.f || image_bw < 4.f) {
            continue;
        }

        // 基于面积约束动态计算输出尺寸(四舍五入到最近整数)
        int32_t output_w = std::lrint(image_bw);
        int32_t output_h = std::lrint(image_bh);
        if (output_h * output_w > final_steps_) {
            if (output_h > ideal_width_ && output_w > ideal_width_) {
                // 情况1: 原矩形足够大, 裁剪理想正方形
                output_h = ideal_width_;
                output_w = final_steps_ / output_h;
            } else if (output_h < ideal_width_) {
                // 情况2: 高度受限 (H < ideal_width)
                output_w = final_steps_ / output_h;
            } else {
                // 情况3: 宽度受限 (W < ideal_width)
                output_h = final_steps_ / output_w;
            }
        }

        results_.push_back({});
        auto &result = results_.back();
        result.score = proposals[row * PROPO_W_ + 4];
        result.label = static_cast<int32_t>(proposals[row * PROPO_W_ + 5]);
        result.bbox = cv::Rect(image_x1, image_y1, output_w, output_h);

        cv::Mat data(result.bbox.height, result.bbox.width, CV_8U, h_final_masks_.data() + row * final_steps_);
        result.mask = data.clone();
        //cv::imwrite(std::format("images_tif/mask_{}_{:03d}.tiff", milliseconds.count(), i), result.mask);

        //cv::Mat image = cv::Mat::zeros(cv::Size(INPUT_W, INPUT_H), CV_8U);
        //image(result.bbox).setTo(255, result.mask);
        //cv::imwrite(std::format("images_png/mask_{}_{:03d}.png", milliseconds.count(), i), image);
    }
    SPDLOG_INFO("TensorRT after postprocess: {}", results_.size());
}