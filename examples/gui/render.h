#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "render-config.h"

namespace example {

class RenderStatistics {
 public:
  RenderStatistics() : num_shoot_rays(0.0), average_traversals(0.0) {}
  ~RenderStatistics() {}

  double num_shoot_rays;
  double average_traversals;
};

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

  /// Builds BVH.
  bool BuildBVH(bool use_sbvh);

  /// Returns false when the rendering was canceled.
  bool Render(RenderStatistics* render_stats, float* rgba, float* aux_rgba,
              int* sample_counts, float quat[4], const RenderConfig& config,
              std::atomic<bool>& cancel_flag);
};
};

#endif  // EXAMPLE_RENDER_H_
