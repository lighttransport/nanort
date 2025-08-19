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

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
//#include "imgui_impl_opengl3_loader.h"

#include "camera.h"
#include "render-config.h"
#include "render.h"
#include "trackball.h"
#include "matrix.h"

#ifdef WIN32
#undef min
#undef max
#endif

#define SHOW_BUFFER_COLOR (0)
#define SHOW_BUFFER_NORMAL (1)
#define SHOW_BUFFER_POSITION (2)
#define SHOW_BUFFER_DEPTH (3)
#define SHOW_BUFFER_TEXCOORD (4)
#define SHOW_BUFFER_VARYCOORD (5)
#define SHOW_BUFFER_VERTEXCOLOR (6)
#define SHOW_BUFFER_MATERIALID (7)

GLFWwindow* gWindow = nullptr;
int gWidth = 512;
int gHeight = 512;
float gMousePosX = 0.0f, gMousePosY = 0.0f;
bool gMouseLeftDown = false;
int gShowBufferMode = SHOW_BUFFER_COLOR;
bool gTabPressed = false;
bool gShiftPressed = false;
bool gCtrlPressed = false;
bool gAltPressed = false;
float gShowPositionScale = 1.0f;
float gShowDepthRange[2] = {0.0f, 10.f};
bool gShowDepthPeseudoColor = true;
float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};

example::Renderer gRenderer;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
example::RenderConfig gRenderConfig;
std::mutex gMutex;

std::vector<float> gDisplayRGBA;
std::vector<float> gRGBA;
std::vector<float> gAuxRGBA;
std::vector<int> gSampleCounts;
std::vector<float> gNormalRGBA;
std::vector<float> gPositionRGBA;
std::vector<float> gDepthRGBA;
std::vector<float> gTexCoordRGBA;
std::vector<float> gVaryCoordRGBA;
std::vector<float> gVertexColorRGBA;
std::vector<int> gMaterialID;

GLuint gTextureID = 0;

// Pitch/Yaw camera control variables
float gPitch = 0.0f; // degrees
float gYaw = 0.0f;   // degrees
const float kPitchMin = -89.0f, kPitchMax = 89.0f;
const float kYawMin = -180.0f, kYawMax = 180.0f;

// Convert pitch/yaw to quaternion
void PitchYawToQuat(float pitch, float yaw, float quat[4]) {
  const float kPI = 3.14159265358979323846f;
  float pitchRad = pitch * float(kPI) / 180.0f;
  float yawRad = yaw * float(kPI) / 180.0f;
  float cy = cosf(yawRad * 0.5f);
  float sy = sinf(yawRad * 0.5f);
  float cp = cosf(pitchRad * 0.5f);
  float sp = sinf(pitchRad * 0.5f);
  // Yaw (Y axis), then Pitch (X axis)
  quat[0] = sp * cy;         // x
  quat[1] = cp * sy;         // y
  quat[2] = -sp * sy;        // z
  quat[3] = cp * cy;         // w
}

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
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto startT = std::chrono::system_clock::now();

    bool initial_pass = false;
    {
      std::lock_guard<std::mutex> guard(gMutex);
      if (gRenderConfig.pass == 0) {
        initial_pass = true;
      }
    }

    gRenderCancel = false;

    bool ret = gRenderer.Render(&gRGBA.at(0), &gAuxRGBA.at(0), &gSampleCounts.at(0),
                               gRenderConfig, gRenderCancel);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);
      gRenderConfig.pass++;
    }

    auto endT = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> ms = endT - startT;
  }
}

void UpdateCameraQuatFromPitchYaw() {
  PitchYawToQuat(gPitch, gYaw, gRenderConfig.quat);
  RequestRender();
}


void InitRender(example::RenderConfig* rc) {
  rc->pass = 0;
  rc->max_passes = 128;

  gSampleCounts.resize(rc->width * static_cast<size_t>(rc->height));
  std::fill(gSampleCounts.begin(), gSampleCounts.end(), 0.0);

  gDisplayRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gDisplayRGBA.begin(), gDisplayRGBA.end(), 0.0);

  gRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gRGBA.begin(), gRGBA.end(), 0.0);

  gAuxRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gAuxRGBA.begin(), gAuxRGBA.end(), 0.0);

  gNormalRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gNormalRGBA.begin(), gNormalRGBA.end(), 0.0);

  gPositionRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gPositionRGBA.begin(), gPositionRGBA.end(), 0.0);

  gDepthRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gDepthRGBA.begin(), gDepthRGBA.end(), std::numeric_limits<float>::infinity());

  gTexCoordRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gTexCoordRGBA.begin(), gTexCoordRGBA.end(), 0.0);

  gVaryCoordRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gVaryCoordRGBA.begin(), gVaryCoordRGBA.end(), 0.0);

  gVertexColorRGBA.resize(rc->width * static_cast<size_t>(rc->height) * 4);
  std::fill(gVertexColorRGBA.begin(), gVertexColorRGBA.end(), 0.0);

  gMaterialID.resize(rc->width * static_cast<size_t>(rc->height));
  std::fill(gMaterialID.begin(), gMaterialID.end(), -1);

  rc->normalImage = &gNormalRGBA.at(0);
  rc->positionImage = &gPositionRGBA.at(0);
  rc->depthImage = &gDepthRGBA.at(0);
  rc->texcoordImage = &gTexCoordRGBA.at(0);
  rc->varycoordImage = &gVaryCoordRGBA.at(0);
  rc->vertexColorImage = &gVertexColorRGBA.at(0);
  rc->materialIDImage = &gMaterialID.at(0);

  gPitch = 0.0f;
  gYaw = 0.0f;
  UpdateCameraQuatFromPitchYaw();

  trackball(gRenderConfig.quat, 0.0f, 0.0f, 0.0f, 0.0f);

  glGenTextures(1, &gTextureID);
  glBindTexture(GL_TEXTURE_2D, gTextureID);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  (void)window;
  (void)scancode;
  (void)mods;

  if (ImGui::GetIO().WantCaptureKeyboard) {
    return;
  }

  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
    gPitch = 0.0f;
    gYaw = 0.0f;
    UpdateCameraQuatFromPitchYaw();
    RequestRender();
  } else if (key == GLFW_KEY_TAB) {
    gTabPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
  } else if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
    gShiftPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
  } else if (key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT) {
    gAltPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
  } else if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
    gCtrlPressed = (action == GLFW_PRESS || action == GLFW_REPEAT);
  }
}

template <typename T>
T saturate(const T& val, const T& minVal, const T& maxVal) {
  return std::max<T>(minVal, std::min<T>(maxVal, val));
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  (void)window;

  if (ImGui::GetIO().WantCaptureMouse) {
    return;
  }
  
  float x = static_cast<float>(xpos);
  float y = static_cast<float>(ypos);

  #if 0
  //std::cout << "cursporPos: wantCapture " << ImGui::GetIO().WantCaptureMouse << "\n";
  
  if (gMouseLeftDown) { // && !ImGui::GetIO().WantCaptureMouse) {
    float w = static_cast<float>(gRenderConfig.width);
    float h = static_cast<float>(gRenderConfig.height);

    float y_offset = gHeight - h;

    if (gCtrlPressed) {
      const float dolly_scale = 1.0 * gRenderConfig.distance / w;
      gRenderConfig.distance += dolly_scale * (gMousePosY - y);
    } else if (gShiftPressed) {
      const float trans_scale = 1.0f * gRenderConfig.distance;
      float r[4][4];
      build_rotmatrix(r, gRenderConfig.quat);
      Matrix::Inverse(r);

      float pan2d[3] = {trans_scale * float(gMousePosX - x) / w,
                        -trans_scale * float(gMousePosY - y) / h, 0.0};
      float pan3d[3];
      Matrix::MultV(pan3d, r, pan2d);
      for (int i = 0; i < 3; i++) gRenderConfig.look_at[i] += pan3d[i];
    } else {
      trackball(gPrevQuat, (2.f * gMousePosX - w) / (float)w,
                (h - 2.f * (gMousePosY - y_offset)) / (float)h,
                (2.f * x - w) / (float)w,
                (h - 2.f * (y - y_offset)) / (float)h);
      add_quats(gPrevQuat, gRenderConfig.quat, gRenderConfig.quat);
    }
    RequestRender();
  }
  #endif

  gMousePosX = (int)x;
  gMousePosY = (int)y;
}

void CameraControl(float x, float y) {
  if (gMouseLeftDown) {
    float w = static_cast<float>(gRenderConfig.width);
    float h = static_cast<float>(gRenderConfig.height);
    float dx = x - gMousePosX;
    float dy = y - gMousePosY;
    if (gCtrlPressed) {
      const float dolly_scale = 1.0 * gRenderConfig.distance / w;
      gRenderConfig.distance += dolly_scale * (gMousePosY - y);
      RequestRender();
    } else if (gShiftPressed) {
      const float trans_scale = 1.0f * gRenderConfig.distance;
      float r[4][4];
      build_rotmatrix(r, gRenderConfig.quat);
      Matrix::Inverse(r);
      float pan2d[3] = {trans_scale * float(gMousePosX - x) / w,
                        -trans_scale * float(gMousePosY - y) / h, 0.0};
      float pan3d[3];
      Matrix::MultV(pan3d, r, pan2d);
      for (int i = 0; i < 3; i++) gRenderConfig.look_at[i] += pan3d[i];
      RequestRender();
    } else {
      float sensitivity = 0.2f;
      gYaw += dx * sensitivity;
      gPitch += dy * sensitivity;
      gPitch = saturate(gPitch, kPitchMin, kPitchMax);
      gYaw = saturate(gYaw, kYawMin, kYawMax);
      UpdateCameraQuatFromPitchYaw();
    }
  }
  gMousePosX = x;
  gMousePosY = y;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  (void)window;
  (void)mods;

  // (1) ALWAYS forward mouse data to ImGui! This is automatic with default backends. With your own backend:
  ImGuiIO& io = ImGui::GetIO(); 
  io.AddMouseButtonEvent(button, (action == GLFW_PRESS));

  //if (ImGui::GetIO().WantCaptureMouse) {
  //  return;
  //}

#if 0
  //if (!ImGui::GetIO().WantCaptureMouse) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      if (action == GLFW_PRESS) {
        gMouseLeftDown = true;
        trackball(gPrevQuat, 0.0f, 0.0f, 0.0f, 0.0f);
      } else {
        gMouseLeftDown = false;
      }
    }
  //}
#endif
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  (void)window;
  glViewport(0, 0, width, height);
  gWidth = width;
  gHeight = height;
}

void IdToCol(float col[3], int mid) {
  if (mid < 0) {
    col[0] = 0.25f;
    col[1] = 0.25f;
    col[2] = 0.25f;
    return;
  }

  float table[8][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                       {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f},
                       {0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f},
                       {1.0f, 1.0f, 1.0f}, {0.5f, 0.5f, 0.5f}};

  int id = mid % 8;

  col[0] = table[id][0];
  col[1] = table[id][1];
  col[2] = table[id][2];
}

inline float pesudoColor(float v, int ch) {
  if (ch == 0) {
    if (v <= 0.5f)
      return 0.f;
    else if (v < 0.75f)
      return (v - 0.5f) / 0.25f;
    else
      return 1.f;
  } else if (ch == 1) {
    if (v <= 0.25f)
      return v / 0.25f;
    else if (v < 0.75f)
      return 1.f;
    else
      return 1.f - (v - 0.75f) / 0.25f;
  } else if (ch == 2) {
    if (v <= 0.25f)
      return 1.f;
    else if (v < 0.5f)
      return 1.f - (v - 0.25f) / 0.25f;
    else
      return 0.f;
  } else {
    return 1.f;
  }
}

bool gUseDepthMinMaxFromImage = false;
float gDepthImageMin = 0.0f, gDepthImageMax = 1.0f;

void UpdateTexture(int width, int height) {
  // Compute depth min/max from image if flag is set
  if (gUseDepthMinMaxFromImage) {
    gDepthImageMin = std::numeric_limits<float>::max();
    gDepthImageMax = std::numeric_limits<float>::lowest();
    for (int i = 0; i < width * height * 4; ++i) {
      if (i % 4 == 3) continue;  // Skip alpha channel

      float v = gDepthRGBA[i];
      if (std::isfinite(v)) {
        if (v < gDepthImageMin) gDepthImageMin = v;
        if (v > gDepthImageMax) gDepthImageMax = v;
      }
    }
    // If no valid values, fallback to default
    if (!std::isfinite(gDepthImageMin) || !std::isfinite(gDepthImageMax)) {
      gDepthImageMin = 0.0f;
      gDepthImageMax = 1.0f;
    }
  }
  std::vector<unsigned char> buf(width * height * 4);
  
  if (gShowBufferMode == SHOW_BUFFER_COLOR) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int src_idx = ((height - 1 - y) * width + x);
        int dst_idx = (y * width + x);
        
        float r = gRGBA[4 * src_idx + 0];
        float g = gRGBA[4 * src_idx + 1];
        float b = gRGBA[4 * src_idx + 2];
        float a = gRGBA[4 * src_idx + 3];
        if (gSampleCounts[src_idx] > 0) {
          r /= static_cast<float>(gSampleCounts[src_idx]);
          g /= static_cast<float>(gSampleCounts[src_idx]);
          b /= static_cast<float>(gSampleCounts[src_idx]);
          a /= static_cast<float>(gSampleCounts[src_idx]);
        }
        buf[4 * dst_idx + 0] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, r * 255.0f)));
        buf[4 * dst_idx + 1] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, g * 255.0f)));
        buf[4 * dst_idx + 2] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, b * 255.0f)));
        buf[4 * dst_idx + 3] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, a * 255.0f)));
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_NORMAL) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          float val = gNormalRGBA[src_idx] * 0.5f + 0.5f;
          buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, val * 255.0f)));
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_POSITION) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          float val = gPositionRGBA[src_idx] * gShowPositionScale;
          buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, val * 255.0f)));
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_DEPTH) {
    float d_min = gUseDepthMinMaxFromImage ? gDepthImageMin : std::min(gShowDepthRange[0], gShowDepthRange[1]);
    float d_max = gUseDepthMinMaxFromImage ? gDepthImageMax : std::max(gShowDepthRange[0], gShowDepthRange[1]);
    float d_diff = fabsf(d_max - d_min);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          float v = (gDepthRGBA[src_idx] - d_min) / d_diff;
          if (gShowDepthPeseudoColor) {
            float val = pesudoColor(v, c);
            buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, val * 255.0f)));
          } else {
            buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, v * 255.0f)));
          }
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TEXCOORD) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, gTexCoordRGBA[src_idx] * 255.0f)));
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VARYCOORD) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, gVaryCoordRGBA[src_idx] * 255.0f)));
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VERTEXCOLOR) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int c = 0; c < 4; c++) {
          int src_idx = ((height - 1 - y) * width + x) * 4 + c;
          int dst_idx = (y * width + x) * 4 + c;
          buf[dst_idx] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, gVertexColorRGBA[src_idx] * 255.0f)));
        }
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_MATERIALID) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int src_idx = (height - 1 - y) * width + x;
        int dst_idx = y * width + x;
        
        float rgb[3];
        IdToCol(rgb, gMaterialID[src_idx]);
        buf[4 * dst_idx + 0] = static_cast<unsigned char>(rgb[0] * 255.0f);
        buf[4 * dst_idx + 1] = static_cast<unsigned char>(rgb[1] * 255.0f);
        buf[4 * dst_idx + 2] = static_cast<unsigned char>(rgb[2] * 255.0f);
        buf[4 * dst_idx + 3] = 255;
      }
    }
  }

  glBindTexture(GL_TEXTURE_2D, gTextureID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, buf.data());
}

int main(int argc, char** argv) {
  std::string config_filename = "config.json";

  if (argc > 1) {
    config_filename = argv[1];
  }

  {
    bool ret = example::LoadRenderConfig(&gRenderConfig, config_filename.c_str());
    if (!ret) {
      fprintf(stderr, "Failed to load [ %s ]\n", config_filename.c_str());
      return -1;
    }
  }

  {
    bool eson_ret = false;
    if (!gRenderConfig.eson_filename.empty()) {
      eson_ret = gRenderer.LoadEsonMesh(gRenderConfig.eson_filename.c_str());
      if (!eson_ret) {
        fprintf(stderr, "Failed to load [ %s ]\n", gRenderConfig.eson_filename.c_str());
      }
    }
    if (!eson_ret) {
      bool obj_ret = gRenderer.LoadObjMesh(gRenderConfig.obj_filename.c_str(), gRenderConfig.scene_scale);
      if (!obj_ret) {
        fprintf(stderr, "Failed to load [ %s ]\n", gRenderConfig.obj_filename.c_str());
        return -1;
      }
      if (!gRenderConfig.eson_filename.empty()) {
        eson_ret = gRenderer.SaveEsonMesh(gRenderConfig.eson_filename.c_str());
        if (!eson_ret) {
          fprintf(stderr, "Failed to save [ %s ]\n", gRenderConfig.eson_filename.c_str());
        }
      }
    }
  }

  gRenderer.BuildBVH();

  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return -1;
  }

  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  gWindow = glfwCreateWindow(1024, 800, "NanoRT GUI", nullptr, nullptr);
  if (gWindow == nullptr) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(gWindow);
  glfwSwapInterval(1);

  glfwSetKeyCallback(gWindow, keyCallback);
  glfwSetCursorPosCallback(gWindow, cursorPosCallback);
  glfwSetMouseButtonCallback(gWindow, mouseButtonCallback);
  glfwSetFramebufferSizeCallback(gWindow, framebufferSizeCallback);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(gWindow, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  InitRender(&gRenderConfig);

  std::thread renderThread(RenderThread);
  RequestRender();

  while (!glfwWindowShouldClose(gWindow)) {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("UI");
    {
      if (ImGui::InputFloat("distance", &gRenderConfig.distance)) {
        RequestRender();
      }
      if (ImGui::InputFloat("FoV", &gRenderConfig.fov, 1.0f, 5.0f, "%.2f")) {
        gRenderConfig.fov = saturate<float>(gRenderConfig.fov, 0.1f, 180.0f);
        RequestRender();
      }

      if (ImGui::DragFloat3("look-at", gRenderConfig.look_at, 0.1f)) {
        RequestRender();
      }

      namespace cr = Camera::Registry;
      if (ImGui::BeginCombo("camera type", cr::cameraTypes[gRenderConfig.cameraTypeSelection].typeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(cr::cameraTypes); ++i) {
          bool is_selected = cr::cameraTypes[gRenderConfig.cameraTypeSelection].typeName == cr::cameraTypes[i].typeName;
          if (ImGui::Selectable(cr::cameraTypes[i].typeName, is_selected)) {
            Camera::setCameraFromIdx(gRenderConfig, i);
            RequestRender();
          }
          if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
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
      ImGui::SameLine();
      ImGui::RadioButton("vertex col", &gShowBufferMode, SHOW_BUFFER_VERTEXCOLOR);
      ImGui::SameLine();
      ImGui::RadioButton("material id", &gShowBufferMode, SHOW_BUFFER_MATERIALID);

      ImGui::InputFloat("show pos scale", &gShowPositionScale);
      ImGui::InputFloat2("show depth range", gShowDepthRange);
      ImGui::Checkbox("show depth pesudo color", &gShowDepthPeseudoColor);
      ImGui::Checkbox("Use depth min/max from image", &gUseDepthMinMaxFromImage);
      if (gUseDepthMinMaxFromImage) {
        ImGui::Text("Depth image min: %.6f", gDepthImageMin);
        ImGui::Text("Depth image max: %.6f", gDepthImageMax);
      }

      if (ImGui::SliderFloat("Pitch", &gPitch, kPitchMin, kPitchMax)) {
        UpdateCameraQuatFromPitchYaw();
      }
      if (ImGui::SliderFloat("Yaw", &gYaw, kYawMin, kYawMax)) {
        UpdateCameraQuatFromPitchYaw();
      }
      if (ImGui::Button("Reset Pitch/Yaw")) {
        gPitch = 0.0f;
        gYaw = 0.0f;
        UpdateCameraQuatFromPitchYaw();
      }
    }
    ImGui::End();

    int display_w, display_h;
    glfwGetFramebufferSize(gWindow, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.0f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    UpdateTexture(gRenderConfig.width, gRenderConfig.height);

    ImGui::Begin("Render"); //, nullptr, ImGuiWindowFlags_NoMove);
    ImGui::ImageButton("image", (void*)(intptr_t)gTextureID, ImVec2(gRenderConfig.width, gRenderConfig.height));

  {
    ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
    ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
    ImVec2 mousePositionRelative = ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x, mousePositionAbsolute.y - screenPositionAbsolute.y);

    // If the emulator has focus, send it mouse button/keyboard events
    if (ImGui::IsItemFocused())
    {
        if (!gMouseLeftDown && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
          gMousePosX = mousePositionRelative.x;
          gMousePosY = mousePositionRelative.y;
        }
        gMouseLeftDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);

        gShiftPressed = ImGui::IsKeyDown(ImGuiKey_LeftShift) ||
                        ImGui::IsKeyDown(ImGuiKey_RightShift);

        gCtrlPressed = ImGui::IsKeyDown(ImGuiKey_LeftCtrl);
        gTabPressed = ImGui::IsKeyDown(ImGuiKey_Tab);

    }
    // When the emulator looses focus, release all buttons
    else
    {
        gMouseLeftDown = false;
        //SetEmulatorLeftMouseDownState(false);
        // ...etc for other mouse buttons

        gTabPressed = false;
        gShiftPressed = false;
        gCtrlPressed = false;
        //for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++)
        //{
        //    SetEmulatorKeyState(i, false);
        //}
    }

    if (ImGui::IsItemHovered())
    {
        CameraControl(mousePositionRelative.x, mousePositionRelative.y);
    }

  }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(gWindow);
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  gRenderCancel = true;
  gRenderQuit = true;
  renderThread.join();

  glDeleteTextures(1, &gTextureID);

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(gWindow);
  glfwTerminate();

  return EXIT_SUCCESS;
}