#ifndef IBL_LOADER_H_
#define IBL_LOADER_H_

#include <string>
#include <vector>

namespace example {

///
/// Loads HDR image.
///
/// @param[out] out_image Loaded HDR image
/// @param[out] out_width Width of loaded HDR image
/// @param[out] out_height Height of loaded HDR image
/// @param[out] out_channels The number of channels in loaded HDR image
///
bool LoadHDRImage(const std::string &filename,
                  std::vector<float> *out_image, int *out_width,
                  int *out_height, int *out_channels);

} // namespace eample


#endif // IBL_LOADER_H_