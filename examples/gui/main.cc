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

#include "imgui.h"
#include "imgui_impl_btgui.h"

b3gDefaultOpenGLWindow* window = 0;
int gWidth = 512;
int gHeight = 512;
int gMousePosX = -1, gMousePosY = -1;

void checkErrors(std::string desc) {
  GLenum e = glGetError();
  if (e != GL_NO_ERROR) {
    fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
    exit(20);
  }
}

void keyboardCallback(int keycode, int state) {
  // printf("hello key %d, state %d\n", keycode, state);
  if (keycode == 27) {
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


  while (!window->requestedExit()) {
    window->startRendering();

    checkErrors("begin frame");

    ImGui_ImplBtGui_NewFrame(gMousePosX, gMousePosY);
    {
      static float col[3] = {0,0,0};
      static float f = 0.0f;
      ImGui::ColorEdit3("color", col);
      ImGui::InputFloat("intensity", &f);
    }

    glViewport(0, 0, window->getWidth(), window->getHeight());
    glClearColor(0, 0.1, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

    checkErrors("clear");

    ImGui::Render();

    checkErrors("im render");

    window->endRendering();
  }

  ImGui_ImplBtGui_Shutdown();
  delete window;

  return EXIT_SUCCESS;
}
