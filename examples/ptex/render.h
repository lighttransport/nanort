#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include "render-config.h"

#include <atomic>  // C++11
#include <string>

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads wavefront .obj mesh.
  bool LoadObjMesh(const char* obj_filename, float scene_scale);

  /// Loads Ptex.
  bool LoadPtex(const std::string &ptex_filename);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, int *sample_counts, float quat[4],
              const RenderConfig& config, std::atomic<bool>& cancel_flag);
};
};

#endif  // EXAMPLE_RENDER_H_
