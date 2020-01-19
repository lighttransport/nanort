#pragma once

#include <cstdint>
#include <string>

namespace example {

bool SaveFloatEXR(const float* pixels, uint32_t width, uint32_t height,
             uint32_t channels, const std::string &outfilename);

bool SaveIntEXR(const int* pixels, uint32_t width, uint32_t height,
                uint32_t channels, const std::string &outfilename);

}  // namespace example
