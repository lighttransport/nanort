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

  // intensity
  float intensity = 1.0f;

  // render pass
  int pass;
  int max_passes;

  // Scene input info
  std::string obj_filename;
  float scene_scale;

  std::string ibl_dirname;
  std::string sh_filename;

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
