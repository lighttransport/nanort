#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "render-config.h"
#include "render-layer.h"

#include "scene.h"

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  ///
  /// Loads wavefront .obj mesh.
  /// Also stores textures and materials to Scene.
  ///
  bool LoadObjMesh(const char* obj_filename, float scene_scale, Scene &scene);

  ///
  /// Builds BVH for a given Scene.
  /// when `fit` was true, Fit camera to BVH's bounding box.
  ///
  bool Build(Scene &scene, RenderConfig &config, const bool fit = true);

  /// Returns false when the rendering was canceled.
  bool Render(const Scene &scene, float quat[4],
              const RenderConfig& config, RenderLayer *layer, std::atomic<bool>& cancel_flag);

  bool QueryDistance(const Scene &scene, const float position[3], const float radius, const float tmin = 0.0f, const float tmax = std::numeric_limits<float>::max());
};
};

#endif  // EXAMPLE_RENDER_H_
