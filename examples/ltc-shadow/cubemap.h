#ifndef CUBEMAP_H_
#define CUBEMAP_H_

#include "image.h"

#include <array>

namespace example {

class Cubemap
{
 public:
  Cubemap(const std::array<Image, 6> &src) {
    for (size_t i = 0; i < 6; i++) {
      // assume width == height
      size_t size = src[i].width;

      // allocate extra row and column
      size_t len = (size + 1) * (size + 1) * 3; // Assume RGB
      faces[i].width = size + 1;
      faces[i].height = size + 1;
      faces[i].channels = 3;
      faces[i].pixels.resize(len);

      for (size_t y = 0; y < size; y++) {
        for (size_t x = 0; x < size; x++) {
          faces[i].pixels[3 * (y * faces[i].width + x) + 0] = src[i].pixels[3 * (y * src[i].width + x) + 0];
          faces[i].pixels[3 * (y * faces[i].width + x) + 1] = src[i].pixels[3 * (y * src[i].width + x) + 1];
          faces[i].pixels[3 * (y * faces[i].width + x) + 2] = src[i].pixels[3 * (y * src[i].width + x) + 2];
        }
      }
  
    }
    
  }

  Cubemap(int size) {
    // allocate extra row and column
    size_t len = (size + 1) * (size + 1) * 3; // Assume RGB
    for (size_t i = 0; i < 6; i++) {
      faces[i].width = size + 1;
      faces[i].height = size + 1;
      faces[i].channels = 3;
      faces[i].pixels.resize(len);
    }
  }

  int dim() const {
    return faces[0].width - 1;
  }

  int bytesPerRow() const {
    return 3 * sizeof(float) * faces[0].width;
  }

  std::array<Image, 6> faces;
};

} // namespace example

#endif // CUBEMAP_H_
