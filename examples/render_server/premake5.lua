sources = {
   "main.cc",
   "civetweb/src/CivetServer.cpp",
   "civetweb/src/civetweb.c"
   }

solution "RenderServerSolution"
   configurations { "Release", "Debug" }

   if os.is("Windows") then
      platforms { "x64", "x32" }
   else
      platforms { "native", "x64", "x32" }
   end

   -- Use c++11
   flags { "c++11" }

   -- A project defines one build target
   project "render_server"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../../" }
      includedirs { "./civetweb/include/" }

      if os.is("Windows") then
         flags { "FatalCompileWarnings" }
         warnings "Extra" -- /W4

         defines { "NOMINMAX" }
         buildoptions { "/W4" } -- raise compile error level.
      end
      if os.is("Linux") then
         links {"pthread", "dl"}
      end
      if os.is("MacOSX") then
      end

      configuration "Debug"
         defines { "DEBUG" } -- -DDEBUG
         symbols "On"
         targetname "render_server_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
	 symbols "On"
	 optimize "On"
         targetname "render_server"
