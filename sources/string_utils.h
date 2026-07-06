#ifndef __INC_STRING_UTILS_H
#define __INC_STRING_UTILS_H

#include <string>


std::string trim(const std::string &s);
std::string trim_l(const std::string &s);
std::string trim_r(const std::string &s);

std::string to_lower(const std::string &str);
std::string to_upper(const std::string &str);

#endif //__INC_STRING_UTILS_H