#include "nvinfer_utils.h"
#include "string_utils.h"


namespace {
const auto fDims = [](std::string split_item) {
    nvinfer1::Dims dim_val{};
    std::stringstream s2(split_item);
    while (std::getline(s2, split_item, ',')) {   // NCHW
        split_item = trim(split_item);
        if (split_item.empty()) continue;
        int32_t dim_value = std::stoi(split_item);
        if (dim_val.nbDims >= 2) {
            dim_value = ((dim_value + 31) / 32) * 32;   // 向上取整.
        }
        dim_val.d[dim_val.nbDims] = dim_value;
        dim_val.nbDims++;
    }
    return dim_val;
};
}

std::map<std::string, std::vector<nvinfer1::Dims>> GetDynDims(const std::string &arg, const std::string &dyn_dims) {
    std::map<std::string, std::vector<nvinfer1::Dims>> dimensions;
    if (!dyn_dims.empty()) {
        std::string split_item;
        std::stringstream ss(dyn_dims);
        while (std::getline(ss, split_item, '#')) {          // name:NCHW;NCHW#name:NCHW;
            const auto pos = split_item.find(':');
            if (pos == std::string::npos) {
                SPDLOG_CRITICAL(std::format("dynamic dimensions not accept: {}: {}", arg, dyn_dims));
                throw std::invalid_argument(std::format("{}: {}", arg, dyn_dims));
            }

            std::vector<nvinfer1::Dims> dims;
            std::string name = trim(split_item.substr(0, pos));
            std::stringstream s1(split_item.substr(pos + 1));
            while (std::getline(s1, split_item, ';')) {       // NCHW;NCHW;NCHW
                nvinfer1::Dims dim_val = fDims(split_item);
                SPDLOG_INFO("accept dynamic dimension: {}:{}", arg, dim_val);
                dims.push_back(dim_val);
            }

            if (name.empty() || dims.size() != 3) {
                SPDLOG_CRITICAL(std::format("dynamic dimensions not accept: {}: {}", split_item, dims.size()));
                throw std::invalid_argument(std::format("dynamic dimensions not accept: {}: {}", arg, dyn_dims));
            }

            dimensions[name] = dims;
        }
    }

    return dimensions;
}

NvLogger &NvLogger::GetInstance() {
    static NvLogger instance_;
    return instance_;
}

void NvLogger::SetSeverity(Severity severity) {
    severity_ = severity;
}

std::string NvLogger::GetSeverity(Severity severity) {
    switch (severity) {
        case Severity::kINTERNAL_ERROR: return "[F]";
        case Severity::kERROR:          return "[E]";
        case Severity::kWARNING:        return "[W]";
        case Severity::kINFO:           return "[I]";
        case Severity::kVERBOSE:        return "[V]";
        default:                        return "[*]";
    }
}

void NvLogger::log(Severity severity, const nvinfer1::AsciiChar *msg) noexcept {
    // suppress info-level messages
    if (severity > severity_) {
        return;
    }

    if (severity <= Severity::kERROR) {
        SPDLOG_ERROR("{}", msg);
    } else if (severity == Severity::kWARNING) {
        SPDLOG_WARN("{}", msg);
    } else if (severity == Severity::kINFO) {
        SPDLOG_INFO("{}", msg);
    } else if (severity == Severity::kVERBOSE) {
        SPDLOG_INFO("{}", msg);
    } else {
        SPDLOG_INFO("{}", msg);
    }
}