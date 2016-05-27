#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic> // C++11

namespace example
{

typedef struct
{
	// framebuffer
  int width;
  int height;

	// camera
	float eye[3];
	float up[3];
	float look_at[3];
	float fov;						// vertical fov in degree.

	// render pass
  int pass;
  int max_passes;

  // For debugging. Array size = width * height * 4.
  float *normalImage;
  float *positionImage;
  float *texcoordImage;

} RenderConfig;

class Renderer
{
 public:
	Renderer() {}
	~Renderer() {}

	bool LoadObjMesh(const char* obj_filename, float scene_scale);

	// Returns false when the rendering was canceled.
	bool Render(float *rgba, float *aux_rgba, const RenderConfig& config, std::atomic<bool>& cancel_flag);

};


};

#endif // EXAMPLE_RENDER_H_
