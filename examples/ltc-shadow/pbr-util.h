#ifndef PBR_UTIL_H_
#define PBR_UTIL_H_

#include <cstdlib>

#include "image.h"
#include "cubemap.h"

namespace example {

///
/// Compute DFG Lut.
/// x axis : NdotV, y axis : roughness
///  
void BuildDFGLut(const bool multiscatter, const size_t width, Image *output);

///
/// Prefilter environment map and build roughness map 
///
/// Input is cubemap, output is longlat format.
///
/// @param[in] num_samples The number of samples for glossy kernel evaluation
/// @param[in] output_base_height Base(lod 0) height of output longlat image.
/// @param[out] output_levals LoD of prefiltered roughness envmap.
///
void BuildPrefilteredRoughnessMap(const Cubemap &cubemap, int num_samples, const size_t output_base_height, std::vector<Image> *output_levels);


bool BuildPrefilteredRoughnessMap(const Image& longlat, int num_samples,
                                  const size_t output_base_height,
                                  std::vector<Image>* output_levels);

} // namespace example


#endif // PBR_UTIL_H_
