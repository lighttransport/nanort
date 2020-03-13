sources = {
   "main.cc",
   "obj-writer.cc",
   "../render-config.cc",
   "../../common/matrix.cc",
   }

-- premake5.lua
solution "FaceSorterCliSolution"
   configurations { "Release", "Debug" }

   if os.is("Windows") then
      platforms { "x64", "x32" }
   else
      platforms { "native", "x64", "x32" }
   end


   -- Use C++11
   flags { "c++11" }

   -- A project defines one build target
   project "facesorter_cli"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../", "../../" }
      includedirs { "../../common/" }

      if os.is("Windows") then
         defines { "NOMINMAX" }
         -- buildoptions { "/openmp" } -- Assume vs2013 or later
         buildoptions { "/W4" } -- raise compile error level.
      end
      if os.is("Linux") then
         -- buildoptions { "-fsanitize=address" }
         -- linkoptions { "-fsanitize=address" }
         links {"pthread", "dl"}
      end
      if os.is("MacOSX") then
         -- Assume brew installed liblas
         includedirs { "/usr/local/include" }
         libdirs  { "/usr/local/lib" }

      end

      configuration "Debug"
         defines { "DEBUG" } -- -DDEBUG
         flags { "Symbols" }
         targetname "facesorter_cli_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
         flags { "Symbols", "Optimize" }
         targetname "facesorter_cli"
