/*
The MIT License (MIT)

Copyright (c) 2015 - 2018 Light Transport Entertainment, Inc.

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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include "imgui.h"
#include "imgui_impl_btgui.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include "render-config.h"
#include "geometry-util.h"
#include "render.h"
#include "save_img.h"
#include "trackball.h"
#include "serialize.h"

#ifdef WIN32
#undef min
#undef max
#endif

#define SHOW_BUFFER_RENDER (0)
#define SHOW_BUFFER_SHADING_NORMAL (1)
#define SHOW_BUFFER_GEOM_NORMAL (2)
#define SHOW_BUFFER_POSITION (3)
#define SHOW_BUFFER_DEPTH (4)
#define SHOW_BUFFER_DIFFUSE (5)
#define SHOW_BUFFER_TEXCOORD (6)
#define SHOW_BUFFER_VARYCOORD (7)
#define SHOW_BUFFER_VERTEXCOLOR (8)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-variable-declarations"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

b3gDefaultOpenGLWindow* window = nullptr;
int gWidth = 512;
int gHeight = 512;
int gMousePosX = -1, gMousePosY = -1;
bool gMouseLeftDown = false;
int gShowBufferMode = SHOW_BUFFER_RENDER;
bool gTabPressed = false;
bool gShiftPressed = false;
float gShowPositionScale = 1.0f;
float gShowDepthRange[2] = {10.0f, 20.f};
bool gShowDepthPeseudoColor = true;
float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
float gPrevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};

example::Renderer gRenderer;
example::Scene gScene;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
example::RenderConfig gRenderConfig;
example::RenderLayer gRenderLayer;
std::mutex gMutex;

std::vector<float> gDisplayRGBA;  // Accumurated image.

int isGammaCorrection = 1;  // For Gamma Carrection
int isGammaCorrectionButton = 1;

char save_filename[256] = "output.exr";
bool isSaveExr = false;

#ifdef __clang__
#pragma clang diagnostic push
#endif

static void RequestRender() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  gRenderRefresh = true;
  gRenderCancel = true;
}

static void RenderThread() {
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
    gRenderLayer.Clear();
  }

  while (1) {
    if (gRenderQuit) return;

    if (!gRenderRefresh || gRenderConfig.pass >= gRenderConfig.max_passes) {
      // Give some cycles to this thread.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    // auto startT = std::chrono::system_clock::now();

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

    bool ret = gRenderer.Render(gScene, gCurrQuat, gRenderConfig, &gRenderLayer,
                                gRenderCancel);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);

      gRenderConfig.pass++;
    }

    // auto endT = std::chrono::system_clock::now();

    // std::chrono::duration<double, std::milli> ms = endT - startT;

    // std::cout << ms.count() << " [ms]\n";
  }
}

static void InitRender(example::RenderConfig* rc) {
  rc->pass = 0;

  gRenderLayer.Resize(size_t(rc->width), size_t(rc->height));

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

static void mouseMoveCallback(float x, float y) {
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
      trackball(gPrevQuat, (2.f * gMousePosX - w) / float(w),
                (h - 2.f * (gMousePosY - y_offset)) / float(h),
                (2.f * x - w) / float(w),
                (h - 2.f * (y - y_offset)) / float(h));
      add_quats(gPrevQuat, gCurrQuat, gCurrQuat);
    }
    RequestRender();
  }

  gMousePosX = int(x);
  gMousePosY = int(y);
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
  // GLfloat h = (GLfloat)height / (GLfloat)width;
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

inline void linerTosRGB(float rgba[]) {
  constexpr float a = 0.055f;
  constexpr float a1 = 1.055f;
  constexpr float p = 1.0f / 2.4f;

  for (size_t i = 0; i < 3; i++) {
    if (rgba[i] <= 0.0031308f) {
      rgba[i] = rgba[i] * 12.92f;
    } else {
      rgba[i] = a1 * std::pow(rgba[i], p) - a;
    }
  }
}

static void Display(int width, int height) {
  std::vector<float> buf(size_t(width * height * 4));
  if (gShowBufferMode == SHOW_BUFFER_RENDER) {
    // normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = gRenderLayer.rgba[4 * i + 0];
      buf[4 * i + 1] = gRenderLayer.rgba[4 * i + 1];
      buf[4 * i + 2] = gRenderLayer.rgba[4 * i + 2];
      buf[4 * i + 3] = gRenderLayer.rgba[4 * i + 3];
      if (gRenderLayer.count[i] > 0) {
        buf[4 * i + 0] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 1] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 2] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 3] /= static_cast<float>(gRenderLayer.count[i]);
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_SHADING_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.shading_normal[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_GEOM_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.geometric_normal[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.position[i] * gShowPositionScale;
    }
  } else if (gShowBufferMode == SHOW_BUFFER_DEPTH) {
    float d_min = std::min(gShowDepthRange[0], gShowDepthRange[1]);
    float d_diff = fabsf(gShowDepthRange[1] - gShowDepthRange[0]);
    d_diff = std::max(d_diff, std::numeric_limits<float>::epsilon());
    for (size_t i = 0; i < buf.size() / 4; i++) {
      float v = (gRenderLayer.depth[i] - d_min) / d_diff;
      if (gShowDepthPeseudoColor) {
        buf[4 * i + 0] = pesudoColor(v, 0);
        buf[4 * i + 1] = pesudoColor(v, 1);
        buf[4 * i + 2] = pesudoColor(v, 2);
        buf[4 * i + 3] = 1.0f;
      } else {
        buf[4 * i + 0] = v;
        buf[4 * i + 1] = v;
        buf[4 * i + 2] = v;
        buf[4 * i + 3] = 1.0f;
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_DIFFUSE) {
    // normalize
    for (size_t i = 0; i < buf.size() / 4; i++) {
      buf[4 * i + 0] = gRenderLayer.diffuse[4 * i + 0];
      buf[4 * i + 1] = gRenderLayer.diffuse[4 * i + 1];
      buf[4 * i + 2] = gRenderLayer.diffuse[4 * i + 2];
      buf[4 * i + 3] = gRenderLayer.diffuse[4 * i + 3];
      if (gRenderLayer.count[i] > 0) {
        buf[4 * i + 0] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 1] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 2] /= static_cast<float>(gRenderLayer.count[i]);
        buf[4 * i + 3] /= static_cast<float>(gRenderLayer.count[i]);
      }
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.texcoord[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VARYCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.varycoord[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_VERTEXCOLOR) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRenderLayer.vertexColor[i];
    }
  }

  if (isSaveExr) {
    save_img::SaveEXR(buf.data(), width, height, save_filename);
    std::cerr << "saved " << save_filename << std::endl;
    isSaveExr = false;
  }

  for (size_t i = 0; i < buf.size() / 4; i++)
    if (isGammaCorrection) linerTosRGB(buf.data() + 4 * i);

  glRasterPos2i(-1, -1);
  glDrawPixels(width, height, GL_RGBA, GL_FLOAT,
               static_cast<const GLvoid*>(&buf.at(0)));
}


static void DrawMaterialUI(std::vector<const char*>& material_names) {
  static int idx = 0;
  ImGui::Begin("Material");

  if (material_names.size() == 0) {
    ImGui::Text("No material in the scene.");
    ImGui::End();
    return;
  }

  ImGui::Combo("materials", &idx, material_names.data(), int(material_names.size()));

  ImGui::End();
}

#ifdef __clang__
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif

static bool ApplyDisplacement(example::Scene &scene, const example::RenderConfig &config)
{
  // find displacement map id(choose the first one)
  uint32_t vdisp_material_id = static_cast<uint32_t>(-1);
  for (size_t i = 0; i < scene.materials.size(); i++) {
    if (scene.materials[i].vdisp_texid >= 0) {
      vdisp_material_id = uint32_t(i);
      std::cout << "Found material with vector displacement : " << scene.materials[i].name << std::endl;
      break;
    }
  }

  if (vdisp_material_id == static_cast<uint32_t>(-1)) {
    std::cerr << "No vector displacement map found." << std::endl;
    return false;
  }

  std::cout << "vdisp matid = " << vdisp_material_id << std::endl;
  assert(vdisp_material_id < uint32_t(scene.materials.size()));
  const example::Material &vdisp_material = scene.materials[vdisp_material_id];

  std::cout << "vdisp texid = " << vdisp_material.vdisp_texid << std::endl;
  assert(vdisp_material.vdisp_texid < int(scene.textures.size()));

  const example::Texture &vdisp_texture = scene.textures[size_t(vdisp_material.vdisp_texid)];

  if (vdisp_texture.components != 3) {
    std::cerr << "Vector displacement texture must be RGB" << std::endl;
    return false;
  }

  example::ApplyVectorDispacement(
    scene.mesh.original_vertices,
    scene.mesh.faces,
    scene.mesh.material_ids,
    scene.mesh.facevarying_uvs,
    scene.mesh.facevarying_normals,
    scene.mesh.facevarying_tangents,
    scene.mesh.facevarying_binormals,
    vdisp_material_id,
    vdisp_texture.image,
    size_t(vdisp_texture.width),
    size_t(vdisp_texture.height),
    int(config.vdisp_space),
    config.vdisp_scale,
    &scene.mesh.displaced_vertices);

  // swap
  scene.mesh.vertices.swap(scene.mesh.displaced_vertices);

  std::vector<float> facevarying_normals;

  std::cout << "Area weighting = " << config.area_weighting << std::endl;

  // Recompute normals
  example::RecomputeSmoothNormals(
    scene.mesh.vertices,
    scene.mesh.faces,
    /* area_weighting */config.area_weighting,
    &facevarying_normals);
    
  // swap
  // TODO(LTE): Save original facevarying normals somewhere
  scene.mesh.facevarying_normals.swap(facevarying_normals);

  std::cout << "Computing tangents and binormals..." << std::endl;

  // Compute tangents and binormals from displaced mesh + recomputed normal.
  example::ComputeTangentsAndBinormals(
    scene.mesh.vertices,
    scene.mesh.faces,
    scene.mesh.facevarying_uvs,
    scene.mesh.facevarying_normals,
    &(scene.mesh.facevarying_tangents),
    &(scene.mesh.facevarying_binormals));

  std::cout << "Finish applying displacements." << std::endl;

  return true;
}

int main(int argc, char** argv) {
  std::string config_filename = "../config.json";

  if (argc > 1) {
    config_filename = argv[1];
  }

  {
    bool ret =
        example::LoadRenderConfig(&gRenderConfig, config_filename.c_str());
    if (!ret) {
      fprintf(stderr, "Failed to load config file [ %s ]\n",
              config_filename.c_str());
      return -1;
    }
  }

  bool eson_load_ok = false;

  if (!gRenderConfig.eson_filename.empty()) {
    eson_load_ok =  LoadSceneFromEson(gRenderConfig.eson_filename, &gScene);
  }

  if (!eson_load_ok) {

    // load obj
    bool obj_ret = gRenderer.LoadObjMesh(gRenderConfig.obj_filename.c_str(),
                                         gRenderConfig.scene_scale, gScene);
    if (!obj_ret) {
      fprintf(stderr, "Failed to load .obj [ %s ]\n",
              gRenderConfig.obj_filename.c_str());
      return -1;
    }

    if (!gRenderConfig.eson_filename.empty()) {
      // serialize to ESON
      if (!SaveSceneToEson(gRenderConfig.eson_filename, gScene)) {
        std::cerr << "Failed to save scene to ESON." << std::endl;
        return -1;
      }

    }
  }

  // save initial vertices to `original_vertices`.
  gScene.mesh.original_vertices = gScene.mesh.vertices;

  // TODO(LTE): Regenerate ESON and BVH when displacement parameter changes.
  ApplyDisplacement(gScene, gRenderConfig);

  bool ret = gRenderer.Build(gScene, gRenderConfig);
  if (!ret) {
    fprintf(stderr, "Failed to build BVH\n");
    return -1;
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

  ImGui::CreateContext();
  ImGui_ImplBtGui_Init(window);

  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("./DroidSans.ttf", 15.0f);

  std::vector<const char*> material_names;
  {
    for (size_t i = 0; i < gScene.materials.size(); i++) {
      if (gScene.materials[i].name.empty()) {
        gScene.materials[i].name = "material_" + std::to_string(i);
        material_names.push_back(gScene.materials[i].name.c_str());
      } else {
        material_names.push_back(gScene.materials[i].name.c_str());
      }
    }
  }

  std::thread renderThread(RenderThread);

  // Trigger initial rendering request
  RequestRender();

  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);
    ImGui::Begin("UI");
    {
      if (ImGui::InputFloat3("eye", gRenderConfig.eye)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("up", gRenderConfig.up)) {
        RequestRender();
      }
      if (ImGui::InputFloat3("look_at", gRenderConfig.look_at)) {
        RequestRender();
      }

      ImGui::RadioButton("render", &gShowBufferMode, SHOW_BUFFER_RENDER);
      ImGui::SameLine();
      ImGui::RadioButton("shading normal", &gShowBufferMode, SHOW_BUFFER_SHADING_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("geom normal", &gShowBufferMode, SHOW_BUFFER_GEOM_NORMAL);
      ImGui::SameLine();
      ImGui::RadioButton("position", &gShowBufferMode, SHOW_BUFFER_POSITION);
      ImGui::SameLine();
      ImGui::RadioButton("depth", &gShowBufferMode, SHOW_BUFFER_DEPTH);
      ImGui::SameLine();
      ImGui::RadioButton("diffuse", &gShowBufferMode, SHOW_BUFFER_DIFFUSE);
      ImGui::SameLine();
      ImGui::RadioButton("texcoord", &gShowBufferMode, SHOW_BUFFER_TEXCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("varycoord", &gShowBufferMode, SHOW_BUFFER_VARYCOORD);
      ImGui::SameLine();
      ImGui::RadioButton("vertex col", &gShowBufferMode,
                         SHOW_BUFFER_VERTEXCOLOR);

      // ImGui::InputFloat("show pos scale", &gShowPositionScale);

      // ImGui::InputFloat2("show depth range", gShowDepthRange);
      // ImGui::Checkbox("show depth pesudo color", &gShowDepthPeseudoColor);

      // if (ImGui::InputFloat("horiz_offset",&gRenderConfig.horiz_offset)) {
      // RequestRender();
      //}

      // if (ImGui::InputInt("tracing depth",&gRenderConfig.tracing_depth)) {
      // RequestRender();
      //}
      if (ImGui::RadioButton("gamma correction", &isGammaCorrectionButton, 1)) {
        isGammaCorrection = !isGammaCorrection;
        isGammaCorrectionButton = isGammaCorrection;
      }
      ImGui::Separator();

      ImGui::InputText("save filename", save_filename, 256);
      if (ImGui::Button("save")) {
        isSaveExr = true;
      }
    }

    ImGui::End();

    DrawMaterialUI(material_names);

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0.0f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    Display(gRenderConfig.width, gRenderConfig.height);

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
  ImGui::DestroyContext();
  delete window;

  return EXIT_SUCCESS;
}
