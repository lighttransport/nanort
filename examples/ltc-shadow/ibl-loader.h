#ifndef IBL_LOADER_H_
#define IBL_LOADER_H_

#include <string>
#include <vector>

#include "material.h"
#include "image.h"
#include "cubemap.h"

namespace example {

///
/// Loads cubemaps
/// Returns loaded levels(LoD) of cubemaps.
///
/// Returns 0 when failed to load any cubemaps.
///
int LoadCubemaps(std::string& dirpath, std::vector<Cubemap> *out_cubemaps);

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

///
/// Loads LDR image. Convert pixel from sRRB space to linear space.
///
bool LoadLDRImage(const std::string &filename,
                  Image *output);

} // namespace eample


#endif // IBL_LOADER_H_
