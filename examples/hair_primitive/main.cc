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

#ifdef __clang__
// Disable some warnings for external files.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wextra-semi"
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

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

#ifdef __clang__
#pragma clang diagnostic pop
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

#ifdef __clang__
// Disable some warnings for external files.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wdisabled-macro-expansion"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wextra-semi"
#endif

#include "imgui.h"
#include "imgui_impl_btgui.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "render-config.h"
#include "render.h"
#include "shader.h"
#include "trackball.h"

typedef enum {
  SHOW_BUFFER_COLOR = 0,
  SHOW_BUFFER_NORMAL,
  SHOW_BUFFER_TANGENT,
  SHOW_BUFFER_POSITION,
  SHOW_BUFFER_DEPTH,
  SHOW_BUFFER_TEXCOORD,
  SHOW_BUFFER_UPARAM,
  SHOW_BUFFER_VPARAM
} ShowBufferMode;

static b3gDefaultOpenGLWindow* window = 0;
static int gWidth = 512;
static int gHeight = 512;
static int gMousePosX = -1, gMousePosY = -1;
static bool gMouseLeftDown = false;
static int gShowBufferMode = SHOW_BUFFER_COLOR;
static bool gTabPressed = false;
static bool gShiftPressed = false;
static float gShowPositionScale = 1.0f;
static float gShowDepthRange[2] = {10.0f, 20.f};
static bool gShowDepthPeseudoColor = true;
static float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
static float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
static float gVParamScale = 1.0f;  // Usually 1/thickness

typedef enum {
  PBRT_INTEGRATOR_PATH = 0,
  PBRT_INTEGRATOR_DIRECT_LIGHTING,
} PbrtIntegrator;

static int gPbrtIntegrator = PBRT_INTEGRATOR_PATH;
static int gPbrtRenderWidth = 128;
static int gPbrtRenderHeight = 128;
static int gPbrtRenderMaxDepth = 10;
static int gPbrtRenderPixelSamples = 64;

static example::Renderer* gRenderer;
static example::RenderConfig* gRenderConfig;

typedef struct
{
  double renderTime_;
  std::mutex mutex_;
  std::atomic<bool> renderQuit_;
  std::atomic<bool> renderRefresh_;
  std::atomic<bool> renderCancel_;
  char pad0;

  example::HairParam hairParam_;

  int pad1;

} UIState;

static UIState *gUIState;

static void RequestRender() {
  {
    std::lock_guard<std::mutex> guard(gUIState->mutex_);
    gRenderConfig->pass = 0;
  }

  gUIState->renderRefresh_ = true;
  gUIState->renderCancel_ = true;
}

static void RenderThread() {
  {
    std::lock_guard<std::mutex> guard(gUIState->mutex_);
    gRenderConfig->pass = 0;
  }

  while (1) {
    if (gUIState->renderQuit_) return;

    if (!gUIState->renderRefresh_ || gRenderConfig->pass >= gRenderConfig->max_passes) {
      // Give some cycles to this thread.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    std::chrono::time_point<std::chrono::system_clock> startT =
        std::chrono::system_clock::now();

    // Initialize display buffer for the first pass.
    bool initial_pass = false;
    {
      std::lock_guard<std::mutex> guard(gUIState->mutex_);
      if (gRenderConfig->pass == 0) {
        initial_pass = true;
      }
    }

    gUIState->renderCancel_ = false;
    // gRenderCancel may be set to true in main loop.
    // Render() will repeatedly check this flag inside the rendering loop.

    bool ret = gRenderer->Render(&gRenderConfig->rgba.at(0),
                                 &gRenderConfig->auxRGBA.at(0),
                                 &gRenderConfig->sampleCounts.at(0), gCurrQuat,
                                 *gRenderConfig, gUIState->renderCancel_);

    if (ret) {
      std::lock_guard<std::mutex> guard(gUIState->mutex_);

      gRenderConfig->pass++;
    }

    std::chrono::time_point<std::chrono::system_clock> endT =
        std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> ms = endT - startT;

    gUIState->renderTime_ = ms.count();
  }
}

static void InitRender(example::RenderConfig* rc) {
  rc->pass = 0;

  rc->max_passes = 128;

  size_t len = static_cast<size_t>(rc->width * rc->height);
  rc->sampleCounts.resize(len, 0.0f);

  rc->displayRGBA.resize(len * 4, 0.0f);
  rc->rgba.resize(len * 4, 0.0f);
  rc->auxRGBA.resize(len * 4, 0.0f);
  rc->normalRGBA.resize(len * 4, 0.0f);
  rc->tangentRGBA.resize(len * 4, 0.0f);
  rc->positionRGBA.resize(len * 4, 0.0f);
  rc->depthRGBA.resize(len * 4, 0.0f);
  rc->texCoordRGBA.resize(len * 4, 0.0f);
  rc->uParamRGBA.resize(len * 4, 0.0f);
  rc->vParamRGBA.resize(len * 4, 0.0f);

  trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
}

static void checkErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

static void keyboardCallback(int keycode, int state) {
  // printf("hello key %d, state %d(ctrl %d)\n", keycode, state,
  //       window->isModifierKeyPressed(B3G_CONTROL));
  // if (keycode == 'q' && window && window->isModifierKeyPressed(B3G_SHIFT)) {
  if (keycode == 27) {
    if (window) window->setRequestExit();
  } else if (keycode == ' ') {
    trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
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

static void mouseMoveCallback(float x, float y) {
  if (gMouseLeftDown) {
    float w = gRenderConfig->width;
    float h = gRenderConfig->height;

    float y_offset = gHeight - h;

    if (gTabPressed) {
      const float dolly_scale = 0.1f;
      gRenderConfig->eye[2] += dolly_scale * (gMousePosY - y);
      gRenderConfig->look_at[2] += dolly_scale * (gMousePosY - y);
    } else if (gShiftPressed) {
      const float trans_scale = 0.02f;
      gRenderConfig->eye[0] += trans_scale * (gMousePosX - x);
      gRenderConfig->eye[1] -= trans_scale * (gMousePosY - y);
      gRenderConfig->look_at[0] += trans_scale * (gMousePosX - x);
      gRenderConfig->look_at[1] -= trans_scale * (gMousePosY - y);

    } else {
      // Adjust y.
      trackball(gPrevQuat, (2.f * gMousePosX - w) / static_cast<float>(w),
                (h - 2.f * (gMousePosY - y_offset)) / static_cast<float>(h),
                (2.f * x - w) / static_cast<float>(w),
                (h - 2.f * (y - y_offset)) / static_cast<float>(h));
      add_quats(gPrevQuat, gCurrQuat, gCurrQuat);
    }
    RequestRender();
  }

  gMousePosX = static_cast<int>(x);
  gMousePosY = static_cast<int>(y);
}

static void mouseButtonCallback(int button, int state, float x, float y) {
  (void)x;
  (void)y;
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

static void resizeCallback(float width, float height) {
  GLfloat h = static_cast<GLfloat>(height) / static_cast<GLfloat>(width);
  GLfloat xmax, znear, zfar;

  (void)h;

  znear = 1.0f;
  zfar = 1000.0f;
  xmax = znear * 0.5f;

  gWidth = static_cast<int>(width);
  gHeight = static_cast<int>(height);
}

static inline float pesudoColor(float v, int ch) {
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

static void Display(int width, int height) {
  std::vector<float> buf(static_cast<size_t>(width * height * 4));
  if (gShowBufferMode == SHOW_BUFFER_COLOR) {
    // normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = gRenderConfig->rgba[4 * i + 0];
      buf[4 * i + 1] = gRenderConfig->rgba[4 * i + 1];
      buf[4 * i + 2] = gRenderConfig->rgba[4 * i + 2];
      buf[4 * i + 3] = gRenderConfig->rgba[4 * i + 3];
      if (gRenderConfig->sampleCounts[i] > 0) {
        buf[4 * i + 0] /= static_cast<float>(gRenderConfig->sampleCounts[i]);
        buf[4 * i + 1] /= static_cast<float>(gRenderConfig->sampleCounts[i]);
        buf[4 * i + 2] /= static_cast<float>(gRenderConfig->sampleCounts[i]);
        buf[4 * i + 3] /= static_cast<float>(gRenderConfig->sampleCounts[i]);
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderConfig->normalRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TANGENT) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderConfig->tangentRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderConfig->positionRGBA[i] * gShowPositionScale;
    }
  } else if (gShowBufferMode == SHOW_BUFFER_DEPTH) {
    float d_min = std::min(gShowDepthRange[0], gShowDepthRange[1]);
    float d_diff = fabsf(gShowDepthRange[1] - gShowDepthRange[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size(); i++) {
      float v = (gRenderConfig->depthRGBA[i] - d_min) / d_diff;
      if (gShowDepthPeseudoColor) {
        buf[i] = pesudoColor(v, i % 4);
      } else {
        buf[i] = v;
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderConfig->texCoordRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_UPARAM) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderConfig->uParamRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VPARAM) {
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = gVParamScale * gRenderConfig->vParamRGBA[4 * i + 0];
      buf[4 * i + 1] = gVParamScale * gRenderConfig->vParamRGBA[4 * i + 1];
      buf[4 * i + 2] = gVParamScale * gRenderConfig->vParamRGBA[4 * i + 2];
      buf[4 * i + 3] = 1.0f;
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

  gRenderConfig = new example::RenderConfig();
  gRenderer = new example::Renderer();
  gUIState = new UIState();

  gRenderer->Init();

  {
    bool ret =
        example::LoadRenderConfig(gRenderConfig, config_filename.c_str());
    if (!ret) {
      std::cerr << "Failed to load config file : " << config_filename.c_str()
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  {
    bool ret = gRenderer->LoadCyHair(
        gRenderConfig->cyhair_filename.c_str(), gRenderConfig->scene_scale,
        gRenderConfig->scene_translate, gRenderConfig->max_strands);
    if (!ret) {
      std::cerr << "Failed to load cyhair : " << gRenderConfig->cyhair_filename
                << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "Loaded hair data " << std::endl;
  }

  {
    bool ret = gRenderer->BuildBVH();
    if (!ret) {
      std::cerr << "Failed to build BVH." << std::endl;
      return EXIT_FAILURE;
    }
  }

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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
  if (!GLEW_VERSION_2_1) {
    fprintf(stderr, "OpenGL 2.1 is not available\n");
    exit(-1);
  }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif

  InitRender(gRenderConfig);

  checkErrors("init");

  window->setMouseButtonCallback(mouseButtonCallback);
  window->setMouseMoveCallback(mouseMoveCallback);
  checkErrors("mouse");
  window->setKeyboardCallback(keyboardCallback);
  checkErrors("keyboard");
  window->setResizeCallback(resizeCallback);
  checkErrors("resize");

  ImGui_ImplBtGui_Init(window);

  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("./DroidSans.ttf", 15.0f);

  std::thread renderThread(RenderThread);

  // Trigger initial rendering request
  RequestRender();

  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);
    ImGui::Begin("UI");
    {
      // static float col[3] = {0, 0, 0};
      // static float f = 0.0f;
      // if (ImGui::ColorEdit3("color", col)) {
      //  RequestRender();
      //}
      // ImGui::InputFloat("intensity", &f);
      if (ImGui::InputFloat3("eye", gRenderConfig->eye)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("up", gRenderConfig->up)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("look_at", gRenderConfig->look_at)) {
        RequestRender();
      }

      ImGui::RadioButton("color", &gShowBufferMode, SHOW_BUFFER_COLOR);
      ImGui::SameLine();
      ImGui::RadioButton("normal", &gShowBufferMode, SHOW_BUFFER_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("tangent", &gShowBufferMode, SHOW_BUFFER_TANGENT);
      ImGui::SameLine();
      ImGui::RadioButton("position", &gShowBufferMode, SHOW_BUFFER_POSITION);
      ImGui::SameLine();
      ImGui::RadioButton("depth", &gShowBufferMode, SHOW_BUFFER_DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("texcoord", &gShowBufferMode, SHOW_BUFFER_TEXCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("u", &gShowBufferMode, SHOW_BUFFER_UPARAM);
      ImGui::SameLine();
      ImGui::RadioButton("v", &gShowBufferMode, SHOW_BUFFER_VPARAM);

      ImGui::InputFloat("show pos scale", &gShowPositionScale);

      ImGui::InputFloat("show v scalee", &gVParamScale);

      ImGui::InputFloat2("show depth range", gShowDepthRange);
      ImGui::Checkbox("show depth pesudo color", &gShowDepthPeseudoColor);

      if (ImGui::TreeNode("hair param"))
      {
        ImGui::DragFloat3("sigma_a", gUIState->hairParam_.sigma_a, 0.01f, 0.0f, 10.0f);
        ImGui::DragFloat("eta", &gUIState->hairParam_.eta, 0.01f, 0.0f, 10.0f);
        ImGui::SliderFloat("beta_m", &gUIState->hairParam_.beta_m, 0.0f, 1.0f);
        ImGui::SliderFloat("beta_n", &gUIState->hairParam_.beta_n, 0.0f, 1.0f);
        ImGui::DragFloat("alpha", &gUIState->hairParam_.alpha, 0.01f, 0.0f, 10.0f);
        ImGui::RadioButton("path", &gPbrtIntegrator, PBRT_INTEGRATOR_PATH);
        ImGui::SameLine();
        ImGui::RadioButton("direct", &gPbrtIntegrator, PBRT_INTEGRATOR_DIRECT_LIGHTING);
        ImGui::PushItemWidth(100);
        ImGui::InputInt("width", &gPbrtRenderWidth);
        ImGui::SameLine();
        ImGui::InputInt("height", &gPbrtRenderHeight);
        ImGui::InputInt("maxdepth", &gPbrtRenderMaxDepth);
        ImGui::SameLine();
        ImGui::InputInt("pixelsamples", &gPbrtRenderPixelSamples);
        ImGui::PopItemWidth();
        ImGui::Button("pbrt render");
        ImGui::TreePop();
      }
    }

    ImGui::End();

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    Display(gRenderConfig->width, gRenderConfig->height);

    ImGui::Render();

    checkErrors("im render");

    window->endRendering();

    // Give some cycles to this thread.
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  printf("quiting...\n");
  {
    gUIState->renderCancel_ = true;
    gUIState->renderQuit_ = true;
    renderThread.join();
  }

  ImGui_ImplBtGui_Shutdown();
  delete window;

  printf("finish\n");
  return EXIT_SUCCESS;
}
