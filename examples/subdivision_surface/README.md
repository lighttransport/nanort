# Subdivision surface example.

Example of ray tracing OpenSubdiv and Embree Subdivision surface.
NanoRT is just used for visualizing tessellated subdivision surface(no direct ray tracing of subdivision surface).

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* git
* cmake
* OpenGL 2.x
* opensubdiv
  * ptex
* embree3

## Setup

### Build Ptex and OpenSubdiv

Build and install Ptex and OpenSubdiv. This example uses CPU backend, thus no need to build GPU backend for OpenSubdiv.

### Build Embree

Or prebuild Embree3 should work.

You can also use `https://github.com/lighttransport/embree-aarch64` to easily build Embree3.

### cmake build

Set `CMAKE_MODULE_PATH` to point to embree build dir.
Set `OSD_DIR` to point to OpenSubdiv install dir.
Set `PTEX_DIR` to point to Ptex install dir.

#### Build on Linux/MacOSX

```
$ rmdir build
$ mkdir build
$ cd build
$ cmake -DCMAKE_MODULE_PATH=/path/to/embree -DOSD_DIR=/path/to/osd -DPTEX_DIR=/path/to/ptex ..
$ make
```

#### Build on Windows

T.B.W.

## Usage

The demo reads geometry data from .obj or .ptx. If you use .ptx, it should have geometry data(ptex can optionally have geometry data)
For example you can download some ptex scene for example from http://ptex.us/samples.html

Geometry data must be composed of quad mesh only.

Edit `config.json` and setup file path to .obj or .ptx, then

```
$ cd build
$ ./ptexrt ../config.json
```

## Note on quad primitive

Quad must be planar.

We tessellate a quad to two-triangles and build BVH, then do ray-triangle intersection.
To compute correct UV, we store

## config.json


* `obj_filename` : filepath : filenam of wavefront obj mesh(.obj)
* `ptex_filename` : filepath : filenam of ptex(.ptx)
* `dump_ptex` : true/false : Dump ptex data after loading it.

When both `ojb_filename` and `ptex_filename` are given, `ptex_filename` will be used.

## TODO

* [ ] Support Quad primitive in ray traversal. http://graphics.cs.kuleuven.be/publications/LD05ERQIT/index.html

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

