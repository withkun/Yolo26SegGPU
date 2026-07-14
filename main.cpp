#include <iostream>
#include <filesystem>

#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "segment_engine.h"
#include "media_reader.h"


// 1. 参数里面有@是按参数索引获取, 比如 @video 不用参数修饰符, 命令行输入: main.exe d:\test.mp4
// 2. 参数里没有@需要加参数修饰符, 比如 path 需要加--修饰符, 命令行输入: main.exe --path d:\test.mp4
const std::string args{
    "{model_yolo    | best.onnx         | model file format of onnx or engine                       }"
    "{input_dims    |                   | dynamic image dimensions as NCHW(image:1,1,960,1280;1,1,1216,1920;1,1,1216,1920)}"
    "{input_file    | images/*.png      | input image file name or pattern                          }"
    "{output_dir    | results/          | result output directory                                   }"
    "{console       | true              | show log console                                          }"
};

// 初始化日志系统
static void slogInit(const bool console) {
    std::vector<spdlog::sink_ptr> sinks;
    try {
        // 循环日志rotating_sink
        constexpr std::string fn("tlvision.log");
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(fn, 50*1024*1024, 100, false));
        // 控制台日志console_sink
        if (console) {
            sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        }
    } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Can not create log file: " << ex.what() << std::endl << "Program exit..." << std::endl;
        std::exit(-1);
    }

    // 创建异步日志
    spdlog::init_thread_pool(64, 1);
    const auto logger = std::make_shared<spdlog::async_logger>("TLA", sinks.begin(), sinks.end(), spdlog::thread_pool());
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    spdlog::flush_on(spdlog::level::info);
    spdlog::flush_every(std::chrono::milliseconds(10));
    spdlog::set_pattern("%^[%Y-%m-%dT%T.%f,%L,%t,%s:%#:%!]%$ %v");
    spdlog::set_level(spdlog::level::info);

    // 打印输出测试
    SPDLOG_INFO("程序启动 ...");
    SPDLOG_INFO("\xE7\xA8\x8B\xE5\xBA\x8F\xE5\x90\xAF\xE5\x8A\xA8 ..."); //程序启动UTF8编码，控制台不应显示乱码
    SPDLOG_INFO("Program started ...");
}

int main(int argc, char **argv) {
    // YoloSegTRT.exe "C:/WORK/YoloSegTRT/yolo11n-seg.onnx" "C:/WORK/YoloSegTRT/dog.jpg"
    // YoloSegTRT.exe "C:/WORK/YoloSegTRT/yolov8n-seg.engine" "C:/WORK/YoloSegTRT/test3.mp4"
    cv::CommandLineParser parser(argc, argv, args);
    const std::string model_yolo(parser.get<std::string>("model_yolo"));
    const std::string input_file(parser.get<std::string>("input_file"));
    const std::string output_dir(parser.get<std::string>("output_dir"));
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_WARNING);
    slogInit(parser.get<bool>("console"));
    SPDLOG_INFO("TensorRT YOLO26🚀实例分割: {}: {}", model_yolo, input_file);

    MediaReader media_reader(input_file);
    if (media_reader.empty()) {
        parser.printMessage();
        std::cerr << "===> TensorRT no image file found: " << input_file << std::endl;
        return -1;
    }
    SPDLOG_INFO("TensorRT total files: {}", media_reader.count());

    // YOLO模型支持一个动态输入: images
    const auto input_dims = GetDynDims("input_dims", parser.get<std::string>("input_dims"));

    SegmentEngine segmentation;
    if (!segmentation.get_engine(model_yolo, input_dims)) {
        parser.printMessage();
        return -1;
    }
    segmentation.create_context();

    const auto frame_n = media_reader.count();
    const auto image_w = media_reader.width();
    const auto image_h = media_reader.height();
    SPDLOG_INFO("===> TensorRT image_sz: {}×{}, count: {}", image_h, image_w, frame_n);
    cv::VideoWriter output("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 24, cv::Size2d(image_w, image_h), true);  // 三通道
    cv::VideoWriter writer("writer.avi", cv::VideoWriter::fourcc('H', '2', '6', '4'), 24, cv::Size2d(image_w, image_h), false); // 单通道
    cv::namedWindow("YOLO26+TensorRT", cv::WINDOW_FREERATIO | cv::WINDOW_GUI_EXPANDED);

    int64_t total_ms = 0;
    for (auto frame : media_reader.frames()) {
        const auto time1 = std::chrono::system_clock::now();

        SegmentResults results = segmentation.RunSync(frame.image);
        if (image_w != frame.image.cols || image_h != frame.image.rows) {
            cv::Mat resized;
            cv::resize(frame.image, resized, cv::Size(image_w, image_h));
            writer.write(resized);
        } else {
            writer.write(frame.image);
        }

        DrawPred(frame.image, results, frame.index);

        const auto time2 = std::chrono::system_clock::now();
        total_ms +=  std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();
        SPDLOG_INFO("TensorRT instance segmentation: {}μs, instances: {}", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count(), results.size());
        cv::putText(frame.image, std::format("Frame: {:d}/{:d} fps: {:d} detect: {:d}", frame.index+1, media_reader.count(), (frame.index+1) * 1000 / total_ms, results.size()),
                    cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::imshow("YOLO26+TensorRT", frame.image);

        if (image_w != frame.image.cols || image_h != frame.image.rows) {
            cv::Mat resized;
            cv::resize(frame.image, resized, cv::Size(image_w, image_h));
            output.write(resized);
        } else {
            output.write(frame.image);
        }

        if (!std::filesystem::exists(std::filesystem::path(output_dir))) {
            std::filesystem::create_directories(output_dir);
        }
        std::filesystem::path path = output_dir / std::filesystem::path(frame.source).filename();
        cv::imwrite(path.string(), frame.image);

        if (cv::waitKey(0) == VK_ESCAPE) {
            break;
        }
    }
    writer.release();
    output.release();

    SPDLOG_INFO("TensorRT run finish. {}ms per image", total_ms / media_reader.count());
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 分割引擎为局部变量, 在函数退出时释放, 释放过程中有日志输出, 此处日志不能关闭.
    //spdlog::shutdown();
    return 0;
}