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

#include <stdio.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

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
#include <atomic>  // C++11
#include <cassert>
#include <chrono>  // C++11
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>  // C++11
#include <string>
#include <thread>  // C++11
#include <vector>

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

struct UIState {
  int width = 512;
  int height = 512;
  int mousePosX = -1, mousePosY = -1;
  bool mouseLeftDown = false;
  bool tabPressed = false;
  bool shiftPressed = false;
  float currQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
  float prevQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
};

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

void RenderThread(UIState* state) {
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

    bool ret = gRenderer.Render(&gRenderLayer, state->currQuat, gRenderConfig,
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

void InitRender(example::RenderConfig* rc, example::RenderLayer* layer,
                UIState* state) {
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

  trackball(state->currQuat, 0.0f, 0.0f, 0.0f, 0.0f);
}

static void glfw_error_callback(int error, const char* description) {
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void glfw_key_callback(GLFWwindow* window, int key, int scancode, int action,
                  int mods) {
  (void)scancode;
  //std::cout << "key " << key << ", scan " << scancode << ", action " << action << ", mods " << mods << "\n";

  ImGuiIO& io = ImGui::GetIO();
  if (io.WantCaptureKeyboard) {
    return;
  }

  if (key == GLFW_KEY_Q && action == GLFW_PRESS && (mods & GLFW_MOD_CONTROL)) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }

  UIState *state = reinterpret_cast<UIState*>(glfwGetWindowUserPointer(window));
  if (!state) {
    std::cerr << "User pointer is not set.\n";
  }

  if ((key == GLFW_KEY_LEFT_SHIFT) || (key == GLFW_KEY_RIGHT_SHIFT)) {
    if (state) {
      state->shiftPressed = (action != GLFW_RELEASE);
    }

  }

  if (key == GLFW_KEY_TAB) {
    if (state) {
      state->tabPressed = (action != GLFW_RELEASE);
    }

  }

  if (key == ' ') {
    trackball(state->currQuat, 0.0f, 0.0f, 0.0f, 0.0f);
    RequestRender();
  }

}

static void glfw_mouse_button_callback(GLFWwindow* window, int button, int action,

                           int mods) {
  (void)mods;

  auto* state = reinterpret_cast<UIState*>(glfwGetWindowUserPointer(window));

  if (state) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
      state->mouseLeftDown = true;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_FALSE)
      state->mouseLeftDown = false;
  }
}

void glfw_cursor_pos_callback(GLFWwindow *window, double mouse_x,
                                double mouse_y) {
  auto state = reinterpret_cast<UIState *>(glfwGetWindowUserPointer(window));
  if (!state) {
    std::cerr << "??? UserPointer not found in cursor_pos_callback.\n";
    return;
  }

  if (!ImGui::GetIO().WantCaptureMouse) {
    if (state->mouseLeftDown) {

      float w = state->width;
      float h = state->height;

      float y_offset = state->height - h;

      if (state->tabPressed) {
        const float dolly_scale = 0.1;
        gRenderConfig.eye[2] += dolly_scale * (state->mousePosY - mouse_y);
        gRenderConfig.look_at[2] += dolly_scale * (state->mousePosY - mouse_y);
      } else if (state->shiftPressed) {
        const float trans_scale = 0.02;
        gRenderConfig.eye[0] += trans_scale * (state->mousePosX - mouse_x);
        gRenderConfig.eye[1] -= trans_scale * (state->mousePosY - mouse_y);
        gRenderConfig.look_at[0] += trans_scale * (state->mousePosX - mouse_x);
        gRenderConfig.look_at[1] -= trans_scale * (state->mousePosY - mouse_y);

      } else {
        // Adjust y.
        trackball(state->prevQuat, (2.f * state->mousePosX - w) / (float)w,
                  (h - 2.f * (state->mousePosY - y_offset)) / (float)h,
                  (2.f * mouse_x - w) / (float)w,
                  (h - 2.f * (mouse_y - y_offset)) / (float)h);
        add_quats(state->prevQuat, state->currQuat, state->currQuat);
      }
      RequestRender();
    }
  }

  state->mousePosX = mouse_x;
  state->mousePosY = mouse_y;
}


#if 0
void checkErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

void keyboardCallback(int keycode, int state) {
  //printf("hello key %d, state %d(ctrl %d)\n", keycode, state,
  //       window->isModifierKeyPressed(B3G_CONTROL));
  
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
      const float trans_scale = 0.02;
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
#endif

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

    // Load .las model
    bool las_ret = gRenderer.LoadLAS(
        gRenderConfig.las_filename.c_str(), gRenderConfig.scene_scale,
        gRenderConfig.radius_scale, gRenderConfig.max_points);
    if (!las_ret) {
      fprintf(stderr, "Failed to load [ %s ]\n",
              gRenderConfig.las_filename.c_str());
      return -1;
    }
  }

  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    std::cerr << "Failed to initialize glfw."
              << "\n";
    return -1;
  }

  // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char* glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

  // Create window with graphics context
  GLFWwindow* window = glfwCreateWindow(
      1280, 720, "Dear ImGui GLFW+OpenGL3 example", nullptr, nullptr);
  if (window == nullptr) return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
  io.ConfigFlags |=
      ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
  ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
  ImGui_ImplOpenGL3_Init(glsl_version);

  gRenderer.BuildBVH();

  UIState ui_state;
  glfwSetWindowUserPointer(window, &ui_state);
  glfwSetKeyCallback(window, glfw_key_callback);
  glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
  glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);

  InitRender(&gRenderConfig, &gRenderLayer, &ui_state);

  ImGui::CreateContext();

  // ImGuiIO& io = ImGui::GetIO();
  //  io.Fonts->AddFontDefault();
  // io.Fonts->AddFontFromFileTTF("./Inconsolata-Regular.ttf", 22.0f);

  std::thread renderThread(RenderThread, &ui_state);

  // Trigger initial rendering request
  RequestRender();

  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  while (!glfwWindowShouldClose(window)) {
    // Poll and handle events (inputs, window resize, etc.)
    // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to
    // tell if dear imgui wants to use your inputs.
    // - When io.WantCaptureMouse is true, do not dispatch mouse input data to
    // your main application, or clear/overwrite your copy of the mouse data.
    // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input
    // data to your main application, or clear/overwrite your copy of the
    // keyboard data. Generally you may always pass all inputs to dear imgui,
    // and hide them from your application based on those two flags.
    glfwPollEvents();
    if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
      ImGui_ImplGlfw_Sleep(10);
      continue;
    }

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

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

    // Rendering
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    Display(gRenderConfig.width, gRenderConfig.height, gRenderConfig,
            gRenderLayer);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);

    // Give some cycles to this thread.
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  printf("quit\n");
  {
    gRenderCancel = true;
    gRenderQuit = true;
    renderThread.join();
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
