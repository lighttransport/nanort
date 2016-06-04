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

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, float quat[4],
              const RenderConfig& config, std::atomic<bool>& cancel_flag);
};
};

#endif  // EXAMPLE_RENDER_H_
