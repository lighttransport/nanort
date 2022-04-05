#ifndef RENDER_CONFIG_H
#define RENDER_CONFIG_H

#include <string>

namespace Camera {
class BaseCamera;
}

namespace example {

struct RenderConfig {
  // framebuffer
  int width;
  int height;

  // camera
  float look_at[3];  // the current look-at point.
  //! the current quaternion for rotation definition.
  float quat[4];
  float distance;  // distance from look_at point.
  float fov;       // vertical fov in degree.

  //! The current selection of the camera in the Camera::gCameraTypes string
  //! array.
  int cameraTypeSelection = 0;
  //! The camera object.
  Camera::BaseCamera *camera = nullptr;

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
};

/// Loads config from JSON file.
bool LoadRenderConfig(example::RenderConfig *config, const char *filename);

}  // namespace example

#endif  // RENDER_CONFIG_H
