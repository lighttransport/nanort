#ifndef EXAMPLE_DENOISE_H_
#define EXAMPLE_DENOISE_H_

#include <vector>

namespace example {

///
/// Denoise given image using simple separable bilateral filter.
///
void Denoise(std::vector<float> &image, int width, int height, int channels, std::vector<float> *output);

};


#endif // EXAMPLE_DENOISE_H_
