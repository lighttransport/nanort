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

#ifdef _MSC_VER
#pragma warning(disable : 4244)
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

#include "ImGuizmo.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4201)
#endif

#include "glm/mat4x4.hpp"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "nanosg.h"
#include "render-config.h"
#include "render.h"
#include "obj-loader.h"
#include "trackball.h"

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#define SHOW_BUFFER_COLOR (0)
#define SHOW_BUFFER_NORMAL (1)
#define SHOW_BUFFER_POSITION (2)
#define SHOW_BUFFER_DEPTH (3)
#define SHOW_BUFFER_TEXCOORD (4)
#define SHOW_BUFFER_VARYCOORD (5)

b3gDefaultOpenGLWindow* window = 0;
int gWidth = 512;
int gHeight = 512;
int gMousePosX = -1, gMousePosY = -1;
bool gMouseLeftDown = false;
int gShowBufferMode = SHOW_BUFFER_COLOR;
bool gTabPressed = false;
bool gShiftPressed = false;
float gShowPositionScale = 1.0f;
float gShowDepthRange[2] = {10.0f, 20.f};
bool gShowDepthPeseudoColor = true;
float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};

static nanosg::Scene<float, example::Mesh<float> > gScene;
static example::Asset gAsset;
static std::vector<nanosg::Node<float, example::Mesh<float> > > gNodes;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
example::RenderConfig gRenderConfig;
std::mutex gMutex;

std::vector<float> gDisplayRGBA;  // Accumurated image.
std::vector<float> gRGBA;
std::vector<float> gAuxRGBA;        // Auxiliary buffer
std::vector<int> gSampleCounts;     // Sample num counter for each pixel.
std::vector<float> gNormalRGBA;     // For visualizing normal
std::vector<float> gPositionRGBA;   // For visualizing position
std::vector<float> gDepthRGBA;      // For visualizing depth
std::vector<float> gTexCoordRGBA;   // For visualizing texcoord
std::vector<float> gVaryCoordRGBA;  // For visualizing varycentric coord

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

    bool ret =
        example::Renderer::Render(&gRGBA.at(0), &gAuxRGBA.at(0), &gSampleCounts.at(0),
                         gCurrQuat, gScene, gAsset, gRenderConfig, gRenderCancel);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);

      gRenderConfig.pass++;
    }

    auto endT = std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> ms = endT - startT;

    // std::cout << ms.count() << " [ms]\n";
  }
}

void InitRender(example::RenderConfig* rc) {
  rc->pass = 0;

  rc->max_passes = 128;

  gSampleCounts.resize(rc->width * rc->height);
  std::fill(gSampleCounts.begin(), gSampleCounts.end(), 0.0);

  gDisplayRGBA.resize(rc->width * rc->height * 4);
  std::fill(gDisplayRGBA.begin(), gDisplayRGBA.end(), 0.0);

  gRGBA.resize(rc->width * rc->height * 4);
  std::fill(gRGBA.begin(), gRGBA.end(), 0.0);

  gAuxRGBA.resize(rc->width * rc->height * 4);
  std::fill(gAuxRGBA.begin(), gAuxRGBA.end(), 0.0);

  gNormalRGBA.resize(rc->width * rc->height * 4);
  std::fill(gNormalRGBA.begin(), gNormalRGBA.end(), 0.0);

  gPositionRGBA.resize(rc->width * rc->height * 4);
  std::fill(gPositionRGBA.begin(), gPositionRGBA.end(), 0.0);

  gDepthRGBA.resize(rc->width * rc->height * 4);
  std::fill(gDepthRGBA.begin(), gDepthRGBA.end(), 0.0);

  gTexCoordRGBA.resize(rc->width * rc->height * 4);
  std::fill(gTexCoordRGBA.begin(), gTexCoordRGBA.end(), 0.0);

  gVaryCoordRGBA.resize(rc->width * rc->height * 4);
  std::fill(gVaryCoordRGBA.begin(), gVaryCoordRGBA.end(), 0.0);

  rc->normalImage = &gNormalRGBA.at(0);
  rc->positionImage = &gPositionRGBA.at(0);
  rc->depthImage = &gDepthRGBA.at(0);
  rc->texcoordImage = &gTexCoordRGBA.at(0);
  rc->varycoordImage = &gVaryCoordRGBA.at(0);

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
    float w = static_cast<float>(gRenderConfig.width);
    float h = static_cast<float>(gRenderConfig.height);

    float y_offset = gHeight - h;

    if (gTabPressed) {
      const float dolly_scale = 0.1f;
      gRenderConfig.eye[2] += dolly_scale * (gMousePosY - y);
      gRenderConfig.look_at[2] += dolly_scale * (gMousePosY - y);
    } else if (gShiftPressed) {
      const float trans_scale = 0.02f;
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

void resizeCallback(float width, float height) {
  //GLfloat h = (GLfloat)height / (GLfloat)width;
  GLfloat xmax, znear, zfar;

  znear = 1.0f;
  zfar = 1000.0f;
  xmax = znear * 0.5f;

  gWidth = static_cast<int>(width);
  gHeight = static_cast<int>(height);
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

void Display(int width, int height) {
  std::vector<float> buf(width * height * 4);
  if (gShowBufferMode == SHOW_BUFFER_COLOR) {
    // normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = gRGBA[4 * i + 0];
      buf[4 * i + 1] = gRGBA[4 * i + 1];
      buf[4 * i + 2] = gRGBA[4 * i + 2];
      buf[4 * i + 3] = gRGBA[4 * i + 3];
      if (gSampleCounts[i] > 0) {
        buf[4 * i + 0] /= static_cast<float>(gSampleCounts[i]);
        buf[4 * i + 1] /= static_cast<float>(gSampleCounts[i]);
        buf[4 * i + 2] /= static_cast<float>(gSampleCounts[i]);
        buf[4 * i + 3] /= static_cast<float>(gSampleCounts[i]);
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gNormalRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gPositionRGBA[i] * gShowPositionScale;
    }
  } else if (gShowBufferMode == SHOW_BUFFER_DEPTH) {
    float d_min = std::min(gShowDepthRange[0], gShowDepthRange[1]);
    float d_diff = fabsf(gShowDepthRange[1] - gShowDepthRange[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size(); i++) {
      float v = (gDepthRGBA[i] - d_min) / d_diff;
      if (gShowDepthPeseudoColor) {
        buf[i] = pesudoColor(v, i % 4);
      } else {
        buf[i] = v;
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gTexCoordRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VARYCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gVaryCoordRGBA[i];
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

  // load config
  {
    bool ret =
        example::LoadRenderConfig(&gRenderConfig, config_filename.c_str());
    if (!ret) {
      std::cerr << "Failed to load [ " << config_filename << " ]" << std::endl;
      return -1;
    }
  }

  // construct the scene
  {
    std::vector<example::Mesh<float> > meshes;
    std::vector<example::Material> materials;
    std::vector<example::Texture> textures;

    bool ret = LoadObj(gRenderConfig.obj_filename, gRenderConfig.scene_scale, &meshes, &materials, &textures);
    if (!ret) {
      std::cerr << "Failed to load .obj [ " << gRenderConfig.obj_filename << " ]" << std::endl;
      return -1;
    }

    gAsset.materials = materials;
    gAsset.textures = textures;

    for (size_t n = 0; n < meshes.size(); n++) {
      size_t mesh_id = gAsset.meshes.size();
      gAsset.meshes.push_back(meshes[mesh_id]);
    }

    for (size_t n = 0; n < gAsset.meshes.size(); n++) {
      nanosg::Node<float, example::Mesh<float> > node(&gAsset.meshes[n]);
      gNodes.push_back(node);
     
      gScene.AddNode(node);
    }

    if (!gScene.Commit()) {
      std::cerr << "Failed to commit the scene." << std::endl;
      return -1;
    }

    float bmin[3], bmax[3];
    gScene.GetBoundingBox(bmin, bmax);
    printf("  # of nodes               : %d\n", int(gNodes.size()));
    printf("  Scene Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
    printf("  Scene Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

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

  if (!GLEW_VERSION_2_1) {
    fprintf(stderr, "OpenGL 2.1 is not available\n");
    exit(-1);
  }
#endif

  InitRender(&gRenderConfig);

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
  //io.Fonts->AddFontFromFileTTF("./DroidSans.ttf", 15.0f);

  glm::mat4 projection(1.0f);
  glm::mat4 view(1.0f);
  glm::mat4 matrix(1.0f);

  std::thread renderThread(RenderThread);

  // Trigger initial rendering request
  RequestRender();

  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);

    ImGuizmo::BeginFrame();
    ImGuizmo::Enable(true);

    //ImGuiIO &io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

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

      ImGui::RadioButton("color", &gShowBufferMode, SHOW_BUFFER_COLOR);
      ImGui::SameLine();
      ImGui::RadioButton("normal", &gShowBufferMode, SHOW_BUFFER_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("position", &gShowBufferMode, SHOW_BUFFER_POSITION);
      ImGui::SameLine();
      ImGui::RadioButton("depth", &gShowBufferMode, SHOW_BUFFER_DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("texcoord", &gShowBufferMode, SHOW_BUFFER_TEXCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("varycoord", &gShowBufferMode, SHOW_BUFFER_VARYCOORD);

      ImGui::InputFloat("show pos scale", &gShowPositionScale);

      ImGui::InputFloat2("show depth range", gShowDepthRange);
      ImGui::Checkbox("show depth pesudo color", &gShowDepthPeseudoColor);
    }

    ImGui::End();

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0.0f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    Display(gRenderConfig.width, gRenderConfig.height);

    // Draw imguizomo
    ImGuizmo::DrawCube(&view[0][0], &projection[0][0], &matrix[0][0]);

    // Draw imgui
    ImGui::Render();

    checkErrors("im render");

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
  delete window;

  return EXIT_SUCCESS;
}
