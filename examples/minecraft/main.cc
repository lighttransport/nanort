/*
The MIT License (MIT)

Copyright (c) 2015 - 2016 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#define USE_OPENGL2
#include "OpenGLWindow/OpenGLInclude.h"
#ifdef _WIN32
#include "OpenGLWindow/Win32OpenGLWindow.h"
#elif defined __APPLE__
#include "OpenGLWindow/MacOpenGLWindow.h"
#else
// assume linux
#include "OpenGLWindow/X11OpenGLWindow.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <atomic>  // C++11
#include <chrono>  // C++11
#include <mutex>   // C++11
#include <thread>  // C++11

#include "imgui.h"
#include "imgui_impl_btgui.h"

#include "render-config.h"
#include "render.h"
#include "trackball.h"

#define SHOW_BUFFER_COLOR (0)
#define SHOW_BUFFER_NORMAL (1)
#define SHOW_BUFFER_POSITION (2)
#define SHOW_BUFFER_DEPTH (3)
#define SHOW_BUFFER_TEXCOORD (4)
#define SHOW_BUFFER_VARYCOORD (5)

struct UIParam {
  int show_buffer_mode;
  float position_scale = 1.0f;
  float depth_range[2];
  bool depth_show_pseudo_color;

  // PBR
  float roughness;
  float metallic;
  float sheen;
  float clearcoat_thickness;
  float clearcoat_roughness;
  float anisotropy;
  float anisotropy_rotation;

  UIParam() {
    show_buffer_mode = SHOW_BUFFER_COLOR;
    position_scale = 1.0f;
    depth_range[0] = 10.0f;
    depth_range[1] = 20.0f;
    depth_show_pseudo_color = true;

    roughness = 0.0;
    metallic = 0.0;
    sheen = 0.0;
    clearcoat_thickness = 0.0;
    clearcoat_roughness = 0.0;
    anisotropy = 0.0;
    anisotropy_rotation = 0.0;
  }
};

UIParam gUIParam;

b3gDefaultOpenGLWindow* window = 0;
int gWidth = 512;
int gHeight = 512;
int gMousePosX = -1, gMousePosY = -1;
bool gMouseLeftDown = false;
bool gTabPressed = false;
bool gShiftPressed = false;
float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};

example::Renderer gRenderer;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
example::RenderConfig gRenderConfig;
example::RenderLayer gRenderLayer;
std::mutex gMutex;

std::vector<float> gDisplayRGBA;  // Accumurated image.

void RequestRender() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  gRenderRefresh = true;
  gRenderCancel = true;
}

void RenderThread() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  while (1) {
    if (gRenderQuit) return;

    if (!gRenderRefresh || gRenderConfig.pass >= gRenderConfig.max_passes) {
      // Give some cycles to this thread.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto startT = std::chrono::system_clock::now();

    // Initialize display buffer for the first pass.
    bool initial_pass = false;
    {
      std::lock_guard<std::mutex> guard(gMutex);
      if (gRenderConfig.pass == 0) {
        initial_pass = true;
      }
    }

    gRenderCancel = false;
    // gRenderCancel may be set to true in main loop.
    // Render() will repeatedly check this flag inside the rendering loop.

    bool ret = gRenderer.Render(&gRenderLayer, gCurrQuat, gRenderConfig,
                                gRenderCancel);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);

      gRenderConfig.pass++;
    }

    auto endT = std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> ms = endT - startT;

    // std::cout << ms.count() << " [ms]\n";
  }
}

void InitRender(example::RenderConfig* rc, example::RenderLayer* layer) {
  rc->pass = 0;

  rc->max_passes = 128;

  layer->width = rc->width;
  layer->height = rc->height;

  layer->sample_counts.resize(rc->width * rc->height);
  std::fill(layer->sample_counts.begin(), layer->sample_counts.end(), 0.0);

  gDisplayRGBA.resize(rc->width * rc->height * 4);
  std::fill(gDisplayRGBA.begin(), gDisplayRGBA.end(), 0.0);

  layer->rgba.resize(rc->width * rc->height * 4);
  std::fill(layer->rgba.begin(), layer->rgba.end(), 0.0);

  layer->normal.resize(rc->width * rc->height * 4);
  std::fill(layer->normal.begin(), layer->normal.end(), 0.0);

  layer->position.resize(rc->width * rc->height * 4);
  std::fill(layer->position.begin(), layer->position.end(), 0.0);

  layer->depth.resize(rc->width * rc->height * 4);
  std::fill(layer->depth.begin(), layer->depth.end(), 0.0);

  layer->texcoord.resize(rc->width * rc->height * 4);
  std::fill(layer->texcoord.begin(), layer->texcoord.end(), 0.0);

  layer->varycoord.resize(rc->width * rc->height * 4);
  std::fill(layer->varycoord.begin(), layer->varycoord.end(), 0.0);

  trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
}

void checkErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

void keyboardCallback(int keycode, int state) {
  printf("hello key %d, state %d(ctrl %d)\n", keycode, state,
         window->isModifierKeyPressed(B3G_CONTROL));
  // if (keycode == 'q' && window && window->isModifierKeyPressed(B3G_SHIFT)) {
  if (keycode == 27) {
    if (window) window->setRequestExit();
  } else if (keycode == ' ') {
    trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
    RequestRender();
  } else if (keycode == 9) {
    gTabPressed = (state == 1);
  } else if (keycode == B3G_SHIFT) {
    gShiftPressed = (state == 1);
  }

  ImGui_ImplBtGui_SetKeyState(keycode, (state == 1));

  if (keycode >= 32 && keycode <= 126) {
    if (state == 1) {
      ImGui_ImplBtGui_SetChar(keycode);
    }
  }
}

void mouseMoveCallback(float x, float y) {
  if (gMouseLeftDown) {
    float w = gRenderConfig.width;
    float h = gRenderConfig.height;

    float y_offset = gHeight - h;

    if (gTabPressed) {
      const float dolly_scale = 0.1;
      gRenderConfig.eye[2] += dolly_scale * (gMousePosY - y);
      gRenderConfig.look_at[2] += dolly_scale * (gMousePosY - y);
    } else if (gShiftPressed) {
      const float trans_scale = 0.01;
      gRenderConfig.eye[0] += trans_scale * (gMousePosX - x);
      gRenderConfig.eye[1] -= trans_scale * (gMousePosY - y);
      gRenderConfig.look_at[0] += trans_scale * (gMousePosX - x);
      gRenderConfig.look_at[1] -= trans_scale * (gMousePosY - y);

    } else {
      // Adjust y.
      trackball(gPrevQuat, (2.f * gMousePosX - w) / (float)w,
                (h - 2.f * (gMousePosY - y_offset)) / (float)h,
                (2.f * x - w) / (float)w,
                (h - 2.f * (y - y_offset)) / (float)h);
      add_quats(gPrevQuat, gCurrQuat, gCurrQuat);
    }
    RequestRender();
  }

  gMousePosX = (int)x;
  gMousePosY = (int)y;
}

void mouseButtonCallback(int button, int state, float x, float y) {
  ImGui_ImplBtGui_SetMouseButtonState(button, (state == 1));

  ImGuiIO& io = ImGui::GetIO();
  if (io.WantCaptureMouse || io.WantCaptureKeyboard) {
    return;
  }

  // left button
  if (button == 0) {
    if (state) {
      gMouseLeftDown = true;
      trackball(gPrevQuat, 0.0f, 0.0f, 0.0f, 0.0f);
    } else
      gMouseLeftDown = false;
  }
}

void resizeCallback(float width, float height) {
  GLfloat h = (GLfloat)height / (GLfloat)width;
  GLfloat xmax, znear, zfar;

  znear = 1.0f;
  zfar = 1000.0f;
  xmax = znear * 0.5f;

  gWidth = width;
  gHeight = height;
}

inline float pesudoColor(float v, int ch) {
  if (ch == 0) {  // red
    if (v <= 0.5f)
      return 0.f;
    else if (v < 0.75f)
      return (v - 0.5f) / 0.25f;
    else
      return 1.f;
  } else if (ch == 1) {  // green
    if (v <= 0.25f)
      return v / 0.25f;
    else if (v < 0.75f)
      return 1.f;
    else
      return 1.f - (v - 0.75f) / 0.25f;
  } else if (ch == 2) {  // blue
    if (v <= 0.25f)
      return 1.f;
    else if (v < 0.5f)
      return 1.f - (v - 0.25f) / 0.25f;
    else
      return 0.f;
  } else {  // alpha
    return 1.f;
  }
}

void Display(int width, int height, const example::RenderConfig& config,
             const example::RenderLayer& layer) {
  std::vector<float> buf(width * height * 4);
  if (gUIParam.show_buffer_mode == SHOW_BUFFER_COLOR) {
    // normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = layer.rgba[4 * i + 0];
      buf[4 * i + 1] = layer.rgba[4 * i + 1];
      buf[4 * i + 2] = layer.rgba[4 * i + 2];
      buf[4 * i + 3] = layer.rgba[4 * i + 3];
      if (layer.sample_counts[i] > 0) {
        buf[4 * i + 0] /= static_cast<float>(layer.sample_counts[i]);
        buf[4 * i + 1] /= static_cast<float>(layer.sample_counts[i]);
        buf[4 * i + 2] /= static_cast<float>(layer.sample_counts[i]);
        buf[4 * i + 3] /= static_cast<float>(layer.sample_counts[i]);
      }
    }
  } else if (gUIParam.show_buffer_mode == SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = layer.normal[i];
    }
  } else if (gUIParam.show_buffer_mode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = layer.position[i] * gUIParam.position_scale;
    }
  } else if (gUIParam.show_buffer_mode == SHOW_BUFFER_DEPTH) {
    float d_min = std::min(gUIParam.depth_range[0], gUIParam.depth_range[1]);
    float d_diff = fabsf(gUIParam.depth_range[1] - gUIParam.depth_range[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size(); i++) {
      float v = (layer.depth[i] - d_min) / d_diff;
      if (gUIParam.depth_show_pseudo_color) {
        buf[i] = pesudoColor(v, i % 4);
      } else {
        buf[i] = v;
      }
    }
  } else if (gUIParam.show_buffer_mode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = layer.texcoord[i];
    }
  } else if (gUIParam.show_buffer_mode == SHOW_BUFFER_VARYCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = layer.varycoord[i];
    }
  }

  glRasterPos2i(-1, -1);
  glDrawPixels(width, height, GL_RGBA, GL_FLOAT,
               static_cast<const GLvoid*>(&buf.at(0)));
}

int main(int argc, char** argv) {
  std::string config_filename = "config.json";

  if (argc > 1) {
    config_filename = argv[1];
  }

#ifdef _OPENMP
  printf("OpenMP cores = %d\n", omp_get_max_threads());
#endif

  {
    bool ret =
        example::LoadRenderConfig(&gRenderConfig, config_filename.c_str());
    if (!ret) {
      fprintf(stderr, "Failed to load [ %s ]\n", config_filename.c_str());
      return -1;
    }

    // Load Minecraft model
    bool las_ret = gRenderer.LoadMI(gRenderConfig.mi_filename.c_str(),
                                    gRenderConfig.scene_scale);
    if (!las_ret) {
      fprintf(stderr, "Failed to load [ %s ]\n",
              gRenderConfig.mi_filename.c_str());
      return -1;
    }
  }

  gRenderer.BuildBVH();

  window = new b3gDefaultOpenGLWindow;
  b3gWindowConstructionInfo ci;
#ifdef USE_OPENGL2
  ci.m_openglVersion = 2;
#endif
  ci.m_width = 1024;
  ci.m_height = 800;
  window->createWindow(ci);

  window->setWindowTitle("view");

#ifndef __APPLE__
#ifndef _WIN32
  // some Linux implementations need the 'glewExperimental' to be true
  glewExperimental = GL_TRUE;
#endif
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    exit(-1);
  }

  if (!GLEW_VERSION_2_1) {
    fprintf(stderr, "OpenGL 2.1 is not available\n");
    exit(-1);
  }
#endif

  InitRender(&gRenderConfig, &gRenderLayer);

  checkErrors("init");

  window->setMouseButtonCallback(mouseButtonCallback);
  window->setMouseMoveCallback(mouseMoveCallback);
  checkErrors("mouse");
  window->setKeyboardCallback(keyboardCallback);
  checkErrors("keyboard");
  window->setResizeCallback(resizeCallback);
  checkErrors("resize");

  ImGui::CreateContext();
  ImGui_ImplBtGui_Init(window);

  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("./Inconsolata-Regular.ttf", 15.0f);

  std::thread renderThread(RenderThread);

  // Trigger initial rendering request
  RequestRender();

  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);
    ImGui::Begin("UI");
    {
      static float col[3] = {0, 0, 0};
      static float f = 0.0f;
      // if (ImGui::ColorEdit3("color", col)) {
      //  RequestRender();
      //}
      // ImGui::InputFloat("intensity", &f);
      if (ImGui::InputFloat3("eye", gRenderConfig.eye)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("up", gRenderConfig.up)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("look_at", gRenderConfig.look_at)) {
        RequestRender();
      }

      ImGui::RadioButton("color", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_COLOR);
      ImGui::SameLine();
      ImGui::RadioButton("normal", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("position", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_POSITION);
      ImGui::SameLine();
      ImGui::RadioButton("depth", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("texcoord", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_TEXCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("varycoord", &gUIParam.show_buffer_mode,
                         SHOW_BUFFER_VARYCOORD);

      ImGui::InputFloat("show pos scale", &gUIParam.position_scale);

      ImGui::InputFloat2("show depth range", gUIParam.depth_range);
      ImGui::Checkbox("show depth pseudo color",
                      &gUIParam.depth_show_pseudo_color);
    }

    ImGui::End();

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0, 0.1, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    Display(gRenderConfig.width, gRenderConfig.height, gRenderConfig,
            gRenderLayer);

    // Draw ImGui
    {
      float fb_scale = window->getRetinaScale();
      glViewport(0, 0, fb_scale * window->getWidth(),
                 fb_scale * window->getHeight());
      ImGui::Render();
      checkErrors("im render");
    }

    window->endRendering();

    // Give some cycles to this thread.
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  printf("quit\n");
  {
    gRenderCancel = true;
    gRenderQuit = true;
    renderThread.join();
  }

  ImGui_ImplBtGui_Shutdown();
  ImGui::DestroyContext();
  delete window;

  return EXIT_SUCCESS;
}
