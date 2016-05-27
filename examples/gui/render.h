#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic> // C++11

namespace example
{

typedef struct
{
  int width;
  int height;

  int pass;

  int max_passes;

  // For debugging. Array size = width * height * 4.
  float *normalImage;
  float *positionImage;
  float *texcoordImage;

} RenderConfig;

// Returns false when the rendering was canceled.
bool Render(float *rgba, float *aux_rgba, const RenderConfig& config, std::atomic<bool>& cancel_flag);

};

#endif // EXAMPLE_RENDER_H_
