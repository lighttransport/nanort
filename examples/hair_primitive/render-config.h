#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>
#include <vector>

namespace example {

class RenderConfig {
 public:
  RenderConfig() {}
  ~RenderConfig() {}

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

  // Render layers
  std::vector<float> displayRGBA;  // Accumurated image.
  std::vector<float> rgba;
  std::vector<float> auxRGBA;       // Auxiliary buffer
  std::vector<int> sampleCounts;    // Sample num counter for each pixel.
  std::vector<float> normalRGBA;    // For visualizing normal
  std::vector<float> tangentRGBA;   // For visualizing hair tangent
  std::vector<float> positionRGBA;  // For visualizing position
  std::vector<float> depthRGBA;     // For visualizing depth
  std::vector<float> texCoordRGBA;  // For visualizing texcoord
  std::vector<float>
      uParamRGBA;  // For visualizing `u` parameter of curve intersection point
  std::vector<float>
      vParamRGBA;  // For visualizing `v` parameter of curve intersection point

  // Scene input info
  std::string cyhair_filename;
  float scene_scale[3];
  float scene_translate[3];
  int max_strands;  // -1 = read all strands
  float thickness;  // -1 = use thickness in cyhair file.
};

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace

#endif  // RENDER_CONFIG_H
