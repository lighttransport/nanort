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
  float *normalImage;
  float *tangentImage;
  float *positionImage;
  float *depthImage;
  float *texcoordImage;
  float *uparamImage;
  float *vparamImage;

  // Scene input info
  std::string cyhair_filename;
  float scene_scale[3];
  float scene_translate[3];
  int max_strands;  // -1 = read all strands
  float thickness;  // -1 = use thickness in cyhair file.

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
