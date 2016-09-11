#include "OpenGLWindow/X11Window.h"

#include <chrono>
#include <thread>

#include <cstdio>

b3gDefaultWindow *window;

void keyboardCallback(int keycode, int state) {
  printf("hello key %d, state %d\n", keycode, state);
  // if (keycode == 'q' && window && window->isModifierKeyPressed(B3G_SHIFT)) {
  if (keycode == 27) {
    if (window) window->setRequestExit();
  }
}

void mouseMoveCallback(float x, float y) {
  printf("move = %f, %f\n", x, y);
}

void mouseButtonCallback(int button, int state, float x, float y) {
  printf("button = %d\n", button);
}

void resizeCallback(float width, float height) {
  printf("resize: %f, %f\n", width, height);
}

int main(int argc, char** argv) {
  window = new b3gDefaultWindow;
  b3gWindowConstructionInfo ci;
  ci.m_width = 1024;
  ci.m_height = 800;
  window->createWindow(ci);

  window->setWindowTitle("view");

  window->setMouseButtonCallback(mouseButtonCallback);
  window->setMouseMoveCallback(mouseMoveCallback);
  window->setKeyboardCallback(keyboardCallback);
  window->setResizeCallback(resizeCallback);

  while (!window->requestedExit()) {
    window->startRendering();


    window->endRendering();

    // Give some cycles to this thread.
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  delete window;

  return 0;
}
