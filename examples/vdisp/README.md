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
$ make
```

## Prepare mesh and material.

Wavefront .obj is supported.

The renderer itself does not tessellate mesh(just displace vertex position for a given mesh), thus user must supply pre-tessellated mesh.

Specify vector displacement map by `disp filename.exr` in .mtl. LDR texture (e.g. png. color space is assumed as linear) and EXR are supported.
Assume vector displacement is defined in world coordinate by default.

if `vn` is given in .obj, the renderer computes tangents and binormals.

## Usage

Edit `config.json`, then go to build direcory.

```
$ ./vdisp
```

Not that `../config.json` will be read by default and specify path to files(.obj, textures, etc) based on working directory.

## How it works

```
Displace():
  for each vertex:
    sample displace value from vertex's texture coordinate.
    displace vertex

RecomputeVertexNormal():
  compute vertex normal with considering area and angle.
```

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

## TODO

* [ ] Fix normal seams.
* [ ] Support quad polygon.
* [ ] Implement better mesh smoothing algoritm for higher quality dispaced mesh generation.
* [ ] Tangent space vector displacement.
* [ ] Tessellate mesh(subdivision using OpenSubdiv)
  * [ ] Catmark subdivision
  * [ ] Adaptive tessellation
* [ ] Per material control of vector displacement property

