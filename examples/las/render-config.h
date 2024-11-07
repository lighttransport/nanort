#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>
#include <vector>

namespace example {

struct RenderLayer {
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

};

struct RenderConfig {
  // framebuffer
  int width{800};
  int height{600};

  // camera
  float eye[3];
  float up[3];
  float look_at[3];
  float fov;  // vertical fov in degree.

  // render pass
  int pass{0};
  int max_passes{1};

  // Scene input info
  std::string las_filename;
  float scene_scale{1.0f};
  float radius_scale{1.0f};
  uint32_t max_points{~0u};

};

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
