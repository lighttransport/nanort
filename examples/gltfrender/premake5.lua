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

-- premake5.lua
solution "glTFRenderSolution"
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
   
   -- Use C++11
   flags { "c++11" }

   -- A project defines one build target
   project "gltfrender"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../../" }

      if os.is("Windows") then
         defines { "NOMINMAX" }
         buildoptions { "/openmp" } -- Assume vs2013 or later
         buildoptions { "/W4" } -- raise compile error level.
         files{
            "OpenGLWindow/Win32OpenGLWindow.cpp",
            "OpenGLWindow/Win32OpenGLWindow.h",
            "OpenGLWindow/Win32Window.cpp",
            "OpenGLWindow/Win32Window.h",
            }
      end
      if os.is("Linux") then
         files {
            "OpenGLWindow/X11OpenGLWindow.cpp",
            "OpenGLWindow/X11OpenGLWindows.h"
            }
         links {"X11", "pthread", "dl"}
      end
      if os.is("MacOSX") then
         links {"Cocoa.framework"}
         files {
                "OpenGLWindow/MacOpenGLWindow.h",
                "OpenGLWindow/MacOpenGLWindow.mm",
               }
      end

      configuration "Debug"
         defines { "DEBUG" } -- -DDEBUG
         flags { "Symbols" }
         targetname "gltfrender_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
         flags { "Symbols", "Optimize" }
         targetname "gltfrender"
