newoption {
   trigger = "with-gtk3nfd",
   description = "Build with native file dialog support(GTK3 required. Linux only)"
}

sources = {
   "main.cc",
   "render.cc",
   "render-config.cc",
   "trackball.cc",
   "matrix.cc",
   "imgui.cpp",
   "imgui_draw.cpp",
   "imgui_impl_btgui.cpp",
   }

-- premake4.lua
solution "ViewerSolution"
   configurations { "Release", "Debug" }

   if os.is("Windows") then
      platforms { "x64", "x32" }
   else
      platforms { "native", "x64", "x32" }
   end


   projectRootDir = os.getcwd() .. "/"
   dofile ("findOpenGLGlewGlut.lua")
   initOpenGL()
   initGlew()

   -- Use c++11
   flags { "c++11" }

   -- A project defines one build target
   project "viwewer"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../../" }
      includedirs { "nativefiledialog/src/include" }

      if os.is("Windows") then
         defines { "NOMINMAX" }
         defines { "USE_NATIVEFILEDIALOG" }
         buildoptions { "/W4" } -- raise compile error level.
         files{
            "OpenGLWindow/Win32OpenGLWindow.cpp",
            "OpenGLWindow/Win32OpenGLWindow.h",
            "OpenGLWindow/Win32Window.cpp",
            "OpenGLWindow/Win32Window.h",
            }
         files { "nativefiledialog/src/nfd_win.cpp" }
      end
      if os.is("Linux") then
         files {
            "OpenGLWindow/X11OpenGLWindow.cpp",
            "OpenGLWindow/X11OpenGLWindows.h"
            }
         links {"X11", "pthread", "dl"}
         if _OPTIONS["with-gtk3nfd"] then
            defines { "USE_NATIVEFILEDIALOG" }
            includedirs { "./nativefiledialog/src/include" }
            files { "nativefiledialog/src/nfd_gtk.c",
                    "nativefiledialog/src/nfd_common.c"
                  }
            buildoptions { "`pkg-config --cflags gtk+-3.0`" }
            linkoptions { "`pkg-config --libs gtk+-3.0`" }
         end
      end
      if os.is("MacOSX") then
         defines { "USE_NATIVEFILEDIALOG" }
         links {"Cocoa.framework"}
         files {
                "OpenGLWindow/MacOpenGLWindow.h",
                "OpenGLWindow/MacOpenGLWindow.mm",
               }
         files { "nativefiledialog/src/nfd_cocoa.m" }
      end

      configuration "Debug"
         defines { "DEBUG" } -- -DDEBUG
         flags { "Symbols" }
         targetname "view_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
         flags { "Symbols", "Optimize" }
         targetname "view"
