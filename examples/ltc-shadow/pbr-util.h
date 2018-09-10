#ifndef PBR_UTIL_H_
#define PBR_UTIL_H_

#include <cstdlib>

#include "image.h"

namespace example {

//
// Compute DFG Lut.
// x axis : NdotV, y axis : roughness
//  
void BuildDFGLut(const bool multiscatter, const size_t width, Image *output);

} // namespace example


#endif // PBR_UTIL_H_
