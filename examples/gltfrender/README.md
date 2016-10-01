# Simple glTF model rendering example with GUI support(bullet3's OpenGLWindow + ImGui).

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* premake5
* OpenGL 2.x

## Limiation

* Geometry only.
* Only single(first) primitive in .gltf.

## Build on Linux/MacOSX

    $ premake5 gmake
    $ make

## Build on MinGW(Experimental)

    $ premake5 gmake
    $ make

## Build on Windows

    > premake5 vs2013

Or 

    > premake5 vs2015

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/gltfrender

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

## Licenses

* btgui3 : zlib license.
* glew : Modified BSD, MIT license.
* picojson : 2-clause BSD license. See picojson.h for more details.
* ImGui : MIT license.
* stb : Public domain. See stb_*.h for more details.

See ../common for more details.

