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

  // draw primitive range(up to # of triangles in the scene)
  // -1 = draw all
  int draw_primitive_range[2] = {0, -1};

  // render pass
  int pass;
  int max_passes;

  // Scene input info
  std::string obj_filename;
  float scene_scale;

  // for face sorter cli.
  // ray org and ray dir for sorting.
  float ray_org[3] = {0.0f, 0.0f, 100.0f};
  float ray_dir[3] = {0.0f, 0.0f, -1.0f};
  // shape id to use(0 = use the first one)
  int shape_id = 0;

} RenderConfig;

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
