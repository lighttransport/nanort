#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include "render-config.h"

#include "Ptexture.h"

#include <atomic>  // C++11
#include <string>

namespace example {

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads wavefront .obj mesh with all quad faces.
  bool LoadObjQuadMesh(const char* obj_filename, float scene_scale);

  /// Loads mesh from Ptex.
  bool LoadPtexMesh(const std::string& ptex_filename, const bool dump = false);

  ///
  /// Subdivide a mesh.
  ///
  /// Do not call this function during rendering.
  /// Should call `BuildBVH` after `Subdivide` before the rendering.
  ///
  /// @param[in] level Subdivision level.
  /// @param[in] dump_obj Dump subdivided mesh as wavefront .obj?
  ///
  void Subdivide(int level, bool dump_obj = false);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, int* sample_counts, float quat[4],
              const RenderConfig& config, std::atomic<bool>& cancel_flag);

  ///
  /// Shade with ptexture
  /// @param[in] filter Filter type enum value. e.g. 4 = bicubic.
  /// @param[in] lerp Interpolate between mipmap levels.
  /// @param[in] sharpness Filter sharpness.
  /// @param[in] noedgeblend Disable cross-face filtering.
  /// @param[in] start_channel Start channel index. Usually 0.
  /// @param[in] channels Texture channels to use. Usually 1, 2 or 3.
  /// @param[in] face_id Face ID
  /// @param[in] u Varycentric U
  /// @param[in] v Varycentric V
  /// @param[in] uw1 U filter width 1
  /// @param[in] vw1 V filter width 1
  /// @param[in] uw2 U filter width 2
  /// @param[in] vw2 V filter width 2
  /// @param[in] width scale factor for filter width
  /// @param[in] blur amoun to add to filter width
  /// @param[out] rgba ptextured color
  ///
  void ShadePtex(int filter, bool lerp, float sharpness, bool noedgeblend,
                 int start_channel, int channels, int face_id, float u, float v,
                 float uw1, float vw1, float uw2, float vw2, float width,
                 float blue, float rgba[4]);

  Ptex::PtexPtr<Ptex::PtexTexture> _ptex;
};
};  // namespace example

#endif  // EXAMPLE_RENDER_H_
