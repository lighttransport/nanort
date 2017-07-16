#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "render-config.h"
#include "nanosg.h"
#include "mesh.h"
#include "material.h"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Returns false when the rendering was canceled.
  static bool Render(float* rgba, float* aux_rgba, int *sample_counts, float quat[4],
              const nanosg::Scene<float, Mesh<float>> &scene, const std::vector<Material> &materials, const std::vector<Texture> &textures, const RenderConfig& config, std::atomic<bool>& cancel_flag);
};
};

#endif  // EXAMPLE_RENDER_H_
