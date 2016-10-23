sources = {
   "main.cc",
   "render.cc",
   "render-config.cc",
   "cyhair_loader.cc",
   "../common/matrix.cc",
   "../common/trackball.cc",
   "../common/imgui/imgui.cpp",
   "../common/imgui/imgui_draw.cpp",
   "../common/imgui/imgui_impl_btgui.cpp",
   }

-- premake4.lua
solution "HairSolution"
   configurations { "Release", "Debug" }

   if os.is("Windows") then
      platforms { "x64", "x32" }
   else
      platforms { "native", "x64", "x32" }
   end


   -- RootDir for OpenGLWindow
   projectRootDir = os.getcwd() .. "/../common/"
   dofile ("../common/findOpenGLGlewGlut.lua")
   initOpenGL()
   initGlew()

   -- Use c++11
   flags { "c++11" }

   -- A project defines one build target
   project "hair"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../../" }
      includedirs { "../common" }
      includedirs { "../common/imgui" }
      includedirs { "../common/nativefiledialog/src/include" }

      if os.is("Windows") then
         defines { "NOMINMAX" }
         buildoptions { "/W4" } -- raise compile error level.
         files{
            "../common/OpenGLWindow/Win32OpenGLWindow.cpp",
            "../common/OpenGLWindow/Win32OpenGLWindow.h",
            "../common/OpenGLWindow/Win32Window.cpp",
            "../common/OpenGLWindow/Win32Window.h",
            }
      end
      if os.is("Linux") then
         buildoptions { "-fsanitize=address" }
         linkoptions { "-fsanitize=address" }
         files {
            "../common/OpenGLWindow/X11OpenGLWindow.cpp",
            "../common/OpenGLWindow/X11OpenGLWindows.h"
            }
         links {"X11", "pthread", "dl"}
      end
      if os.is("MacOSX") then
         links {"Cocoa.framework"}
         files {
                "../common/OpenGLWindow/MacOpenGLWindow.h",
                "../common/OpenGLWindow/MacOpenGLWindow.mm",
               }
      end

      configuration "Debug"
         defines { "DEBUG" } -- -DDEBUG
         flags { "Symbols" }
         targetname "hair_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
         flags { "Symbols", "Optimize" }
         targetname "hair"
