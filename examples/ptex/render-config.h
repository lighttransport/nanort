#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>

namespace example {

typedef struct {
  // framebuffer
  int width;
  int height;

  // camera
  float eye[3];
  float up[3];
  float look_at[3];
  float fov;  // vertical fov in degree.

  // render pass
  int pass;
  int max_passes;

  // For debugging. Array size = width * height * 4.
  // TODO(LTE): Move to other struct.
  float *normalImage;
  float *positionImage;
  float *depthImage;
  float *texcoordImage;
  float *varycoordImage;
  float *tri_varycoordImage;
  float *vertexColorImage;
  int *faceIdImage; // scalar

  // Scene input info
  std::string obj_filename;
  std::string ptex_filename;
  float scene_scale;

  // Ptex options
  bool dump_ptex = false;

  // Ptex filtering options
  int ptex_filter = 4; // bilinear(see `main.cc` Combo menu for details)
  bool ptex_lerp = false; // interp between mipmap levels.
  float ptex_sharpness = 1.0f; // filter sharpness
  bool ptex_noedgeblend = false; // disable cross-face filtering(true = good for debugging);

  float ptex_uw1 = 0.0f; // [0, 1]
  float ptex_uw2 = 0.0f; // [0, 1]
  float ptex_vw1 = 0.0f; // [0, 1]
  float ptex_vw2 = 0.0f; // [0, 1]
  float ptex_width = 1.0f; // scale factgor for filter width
  float ptex_blur = 0.0f; // [0, 1]
  int ptex_start_channel = 0; // Channel index to start[0, 4]
  int ptex_channels = 3; // 1 = grayscale, 3 == rgb

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
