#ifndef RENDER_LAYER_H_
#define RENDER_LAYER_H_

#include <vector>

namespace example {

struct RenderLayer {
  std::vector<float> displayRGBA;  // Accumurated image.
  std::vector<float> rgba;
  std::vector<float> auxRGBA;        // Auxiliary buffer
  std::vector<int> sampleCounts;     // Sample num counter for each pixel.
  std::vector<float> normalRGBA;     // For visualizing normal
  std::vector<float> positionRGBA;   // For visualizing position
  std::vector<float> depthRGBA;      // For visualizing depth
  std::vector<float> texCoordRGBA;   // For visualizing texcoord
  std::vector<float> varyCoordRGBA;  // For visualizing varycentric coord
};

} // namespace example

#endif // RENDER_LAYER_H_

