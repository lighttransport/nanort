#ifndef EXAMPLE_IMAGE_H_
#define EXAMPLE_IMAGE_H_

#include <vector>

namespace example {

struct Image {
  int width;
  int height;
  int channels; // 3 or 4.
  std::vector<float> pixels;  // channels x width x height
};



} // namespace example

#endif // EXAMPLE_IMAGE_H_
