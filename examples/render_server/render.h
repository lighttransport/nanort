#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "json.hpp"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads wavefront .obj mesh.
  bool LoadObjMesh(const char* obj_filename, float scene_scale);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, int *sample_counts, float quat[4],
              const nlohmann::json& config, std::atomic<bool>& cancel_flag);
};
};

#endif // EXAMPLE_RENDER_H_
