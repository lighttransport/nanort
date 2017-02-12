#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11
#include <vector>

#include "render-config.h"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads partio data.
  bool LoadPartio(const char* partio_filename, const float scene_scale,
                  const float constant_radius = 1.0f);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, int* sample_counts, float quat[4],
              const RenderConfig& config, std::atomic<bool>& cancel_flag);

  std::vector<float> vertices_;  // XYZ
  std::vector<float> radiuss_;   // Scalar
};
};

#endif  // EXAMPLE_RENDER_H_
