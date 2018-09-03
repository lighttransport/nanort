# PBR Surface

Implement in software PBR shading

Code for PBR shading is based on this example from khronos (fragment shader implementation from WebGL glTF reference implementation) https://github.com/KhronosGroup/glTF-WebGL-PBR/blob/master/shaders/pbr-frag.glsl

main() function will load a .glTF file thanks to the glTF library https://github.com/syoyo/tinygltf

![damaged helmet](./helmet.png)

## Build

Use CMake to configure a project for your favorite toolchain.

Here is an example procedure of CMake build.

```
$ mkdir build
$ cmake ..
$ make
```

## Third party libraries

This project uses STB Image and STB Image Write for image file manipulation, and glm for reimplementing GLSL maths.

## Outline

This program contains a simple implementation of a PBR shader rendered with raytracing feature.

This is a simple demonstration program to work on PBR implementation in isolation.

The PbrMaterial struct in main.cc holds the parameters for a PBR "meteal/roughness" material.

- metalness : 0 -> dielectric; 1 -> metalic
- roughness : 0 -> glossy; 1 -> rough.
- albedo : r,g,b color of the material. Also known as "colorFactor" or "baseColor" in other renderer/engines

This example implements really simple point lights as light sources. They only have a spatial position and a color.

The pbr_maths::PbrShader class represent an implementation of the fragment shader from the link above adapted to C++. The class enclose as methods all functions defined in the shader, and all global variables as public attributes.

The program will simply render a quad (made of two triangles) in front of the camera, with the caracteristics of the material. The output image will be "out.bmp" in the working directory.

This is a super simple implementation on top of the minial example to get NanoRT up and running.

