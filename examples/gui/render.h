#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "render-config.h"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads wavefront .obj mesh.
  bool LoadObjMesh(const char* obj_filename, float scene_scale);

  /// Saves .eson mesh.
  bool SaveEsonMesh(const char* eson_filename);

  /// Loads cached .eson mesh.
  bool LoadEsonMesh(const char* eson_filename);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, int* sample_counts,
              const RenderConfig& config, std::atomic<bool>& cancel_flag);
};
}  // namespace example

#endif  // EXAMPLE_RENDER_H_
