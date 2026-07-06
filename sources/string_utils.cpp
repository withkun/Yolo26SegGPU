#include "string_utils.h"
#include <algorithm>


// 常见空白字符(C风格isspace范围)
static constexpr std::string WHITESPACE = " \t\n\r\f\v";

std::string trim(const std::string &s) {
    if (s.empty()) return s;

    const size_t s_idx = s.find_first_not_of(WHITESPACE);
    if (s_idx == std::string::npos) return ""; // 全空白

    const size_t e_idx = s.find_last_not_of(WHITESPACE);
    return s.substr(s_idx, (e_idx - s_idx) + 1);
}

std::string trim_l(const std::string &s) {
    if (s.empty()) return s;

    const size_t s_idx = s.find_first_not_of(WHITESPACE);
    if (s_idx == std::string::npos) return ""; // 全空白

    return s.substr(s_idx);
}

std::string trim_r(const std::string &s) {
    if (s.empty()) return s;

    const size_t e_idx = s.find_last_not_of(WHITESPACE);
    return s.substr(0, e_idx + 1);
}

std::string to_lower(const std::string &str) {
    std::string lower_str = str;
    std::ranges::transform(lower_str, lower_str.begin(), [](uint8_t c) { return std::tolower(c); });
    return lower_str;
}

std::string to_upper(const std::string &str) {
    std::string lower_str = str;
    std::ranges::transform(lower_str, lower_str.begin(), [](uint8_t c) { return std::tolower(c); });
    return lower_str;
}