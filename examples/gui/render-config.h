#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>

namespace example {

typedef struct {
  // framebuffer
  int width;
  int height;

  // camera
  float look_at[3];
  float distance; // distance from look_at
  float fov;  // vertical fov in degree.

  // render pass
  int pass;
  int max_passes;

  // For debugging. Array size = width * height * 4.
  float *normalImage;
  float *positionImage;
  float *depthImage;
  float *texcoordImage;
  float *varycoordImage;
  float *vertexColorImage;
  int *materialIDImage;

  // Scene input info
  std::string obj_filename;
  std::string eson_filename;
  float scene_scale;

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
