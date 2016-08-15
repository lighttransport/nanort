#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>
#include <vector>

namespace example {

typedef struct {
  // color
  std::vector<float> rgba;

  // Stores # of samples for each pixel.
  std::vector<int> sample_counts;

  // For debugging. Array size = width * height * 4.
  std::vector<float> normal;
  std::vector<float> position;
  std::vector<float> depth;
  std::vector<float> texcoord;
  std::vector<float> varycoord;

  int width;
  int height;

} RenderLayer;

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

  // Scene input info
  std::string gltf_filename;
  float scene_scale;

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
