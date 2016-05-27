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

#include "CommonInterfaces/CommonRenderInterface.h"

#ifdef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cassert>
#include <iostream>
#include <vector>

#include <thread> // C++11
#include <atomic> // C++11
#include <chrono> // C++11
#include <mutex> // C++11

#include "imgui.h"
#include "imgui_impl_btgui.h"

#include "render.h"

#define SHOW_BUFFER_COLOR     (0)
#define SHOW_BUFFER_NORMAL    (1)
#define SHOW_BUFFER_POSITION  (2)
#define SHOW_BUFFER_TEXCOORD  (3)

b3gDefaultOpenGLWindow* window = 0;
int gWidth = 512;
int gHeight = 512;
int gMousePosX = -1, gMousePosY = -1;
int gShowBufferMode = SHOW_BUFFER_COLOR;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
example::RenderConfig gRenderConfig;
std::mutex gMutex;

std::vector<float> gRGBA;
std::vector<float> gAuxRGBA;      // Auxiliary buffer  
std::vector<float> gNormalRGBA;   // For visualizing normal 
std::vector<float> gPositionRGBA; // For visualizing position  
std::vector<float> gTexCoordRGBA; // For visualizing texcoord

void RequestRender()
{
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  gRenderRefresh = true;
  gRenderCancel = true;

}

void RenderThread()
{
  {
    std::lock_guard<std::mutex> guard(gMutex);
    gRenderConfig.pass = 0;
  }

  while (1) {
    if (gRenderQuit) return;

    if( !gRenderRefresh || gRenderConfig.pass >= gRenderConfig.max_passes )
    {
        // Give some cycles to this thread.
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
        continue;
    }

    auto startT = std::chrono::system_clock::now();

    gRenderCancel = false;
    // gRenderCancel may be set to true in main loop.
    // Render() will repeatedly check this flag inside the rendering loop.
    bool ret = example::Render(&gRGBA.at(0), &gAuxRGBA.at(0), gRenderConfig, gRenderCancel);

    if (ret) {
      std::lock_guard<std::mutex> guard(gMutex);
      gRenderConfig.pass++;
    }

    auto endT = std::chrono::system_clock::now();

    std::chrono::duration<double, std::milli> ms = endT - startT;

    std::cout << ms.count() << " [ms]\n";

  }
    
}

void InitRenderConfig(example::RenderConfig* rc)
{
  rc->width = 512;
  rc->height = 512;
  rc->pass = 0;
  
  rc->max_passes = 128;

  gRGBA.resize(rc->width * rc->height * 4);
  std::fill(gRGBA.begin(), gRGBA.end(), 0.0);

  gAuxRGBA.resize(rc->width * rc->height * 4);
  std::fill(gAuxRGBA.begin(), gAuxRGBA.end(), 0.0);

  gNormalRGBA.resize(rc->width * rc->height * 4);
  std::fill(gNormalRGBA.begin(), gNormalRGBA.end(), 0.0);

  gPositionRGBA.resize(rc->width * rc->height * 4);
  std::fill(gPositionRGBA.begin(), gPositionRGBA.end(), 0.0);

  gTexCoordRGBA.resize(rc->width * rc->height * 4);
  std::fill(gTexCoordRGBA.begin(), gTexCoordRGBA.end(), 0.0);

  rc->normalImage = &gNormalRGBA.at(0);
  rc->positionImage = &gPositionRGBA.at(0);
  rc->texcoordImage = &gTexCoordRGBA.at(0);
}

void checkErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

void keyboardCallback(int keycode, int state) {
  printf("hello key %d, state %d(ctrl %d)\n", keycode, state, window->isModifierKeyPressed(B3G_CONTROL));
  //if (keycode == 'q' && window && window->isModifierKeyPressed(B3G_SHIFT)) {
  if (keycode == 27 ) {
    if (window) window->setRequestExit();
  }

  ImGui_ImplBtGui_SetKeyState(keycode, (state == 1));

  if (keycode >= 32 && keycode <= 126) {
      if (state == 1) {
        ImGui_ImplBtGui_SetChar(keycode);
      }
  }
}

void mouseMoveCallback(float x, float y) {
  // printf("Mouse Move: %f, %f\n", x, y);

  gMousePosX = (int)x;
  gMousePosY = (int)y;
}

void mouseButtonCallback(int button, int state, float x, float y) {
  ImGui_ImplBtGui_SetMouseButtonState(button, (state == 1));
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

void Display( int width, int height )
{
  std::vector<float> buf(width * height * 4);
  if (gShowBufferMode == SHOW_BUFFER_COLOR) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gRGBA[i] + gAuxRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_NORMAL) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gNormalRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_POSITION) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gPositionRGBA[i];
    }
  } else if (gShowBufferMode == SHOW_BUFFER_TEXCOORD) {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = gTexCoordRGBA[i];
    }
  }
 
  glRasterPos2i( -1, -1 );
  glDrawPixels( width, height, GL_RGBA, GL_FLOAT, static_cast<const GLvoid*>( &buf.at(0) ) );
}

int main(int argc, char** argv) {

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
  //some Linux implementations need the 'glewExperimental' to be true
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

  InitRenderConfig(&gRenderConfig);

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
  //io.Fonts->AddFontDefault();
  io.Fonts->AddFontFromFileTTF("./DroidSans.ttf", 15.0f);


  std::thread renderThread(RenderThread);

  // Trigger initial rendering request
  RequestRender();

  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);
    ImGui::Begin("UI");
    {
      static float col[3] = {0,0,0};
      static float f = 0.0f;
      if (ImGui::ColorEdit3("color", col)) {
        RequestRender();
      }
      ImGui::InputFloat("intensity", &f);

      ImGui::RadioButton("color", &gShowBufferMode, SHOW_BUFFER_COLOR); ImGui::SameLine();
      ImGui::RadioButton("normal", &gShowBufferMode, SHOW_BUFFER_NORMAL); ImGui::SameLine();
      ImGui::RadioButton("position", &gShowBufferMode, SHOW_BUFFER_POSITION); ImGui::SameLine();
      ImGui::RadioButton("texcoord", &gShowBufferMode, SHOW_BUFFER_TEXCOORD);
    }
    ImGui::End();

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0, 0.1, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    Display(gRenderConfig.width, gRenderConfig.height);

    ImGui::Render();

    checkErrors("im render");

    window->endRendering();

    // Give some cycles to this thread.
    std::this_thread::sleep_for( std::chrono::milliseconds( 16 ) );
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
