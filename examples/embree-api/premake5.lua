newoption {
   trigger = "with-gtk3nfd",
   description = "Build with native file dialog support(GTK3 required. Linux only)"
}

newoption {
   trigger = "asan",
   description = "Enable Address Sanitizer(gcc5+ ang clang only)"
}

sources = {
   "main.cc",
   "nanort-embree.cc",
   }

solution "EmbreeAPISolution"
   configurations { "Release", "Debug" }

   if os.is("Windows") then
      platforms { "x64", "x32" }
   else
      platforms { "native", "x64", "x32" }
   end

   -- A project defines one build target
   project "embree-api"
      kind "ConsoleApp"
      language "C++"
      files { sources }

      includedirs { "./", "../../" }
      includedirs { "./include" }
      includedirs { "../nanosg" }
      includedirs { "../common" }
      includedirs { "../common/imgui" }
      includedirs { "../common/glm" }
      --includedirs { "../common/nativefiledialog/src/include" }

      toolset "clang"
      buildoptions { "-Weverything -Werror" }

      if _OPTIONS['asan'] then
         buildoptions { "-fsanitize=address" }
         linkoptions { "-fsanitize=address" }
      end

      if os.is("Windows") then
         flags { "FatalCompileWarnings" }
         warnings "Extra" -- /W4

         defines { "NOMINMAX" }
         defines { "_CRT_SECURE_NO_WARNINGS" }
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
         targetname "render_debug"

      configuration "Release"
         -- defines { "NDEBUG" } -- -NDEBUG
         symbols "On"
	 optimize "On"
         targetname "render"
