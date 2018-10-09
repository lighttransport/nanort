# Vector displacement experiment

## Features

* [ ] Render mesh with vector displacement.
* [ ] Bake vector displaced mesh to normal map using ray tracing.

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* C++11 compiler
* OpenGL 2.x
* CMake

## Build with CMake

```
$ mkdir build
$ cd build
$ cmake ..
$ cmake
```

## Prepare mesh and material.

Wavefront .obj is supported.

The renderer itself does not tessellate mesh(just displace vertex position for a given mesh), thus user must supply pre-tessellated mesh.

For vector displacemt map, please specify `vdisp filename.exr` in .mtl. LDR texture (e.g. png) is supported.
Assume vector displacement is defined in world coordinate by default.

if `vn` is given in .obj, the renderer computes tangents and binormals.

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/vdisp

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

## TODO

* [ ] Tessellate mesh(subdivision using OpenSubdiv)
* [ ] Per material control of vector displacement property

