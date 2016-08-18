# Simple Physically Based Rendering example with GUI support(bullet3's OpenGLWindow + ImGui).

W.I.P.(No PBR yet...)

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* premake5
* OpenGL 2.x

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

    $ ./bin/native/Release/pbr

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

