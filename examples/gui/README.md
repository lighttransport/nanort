# Simple GUI example with bullet3's OpenGLWindow + ImGui.

![](screenshot/nanort_gui.png)

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

`GUISolution.sln` is pre-generated solution file for Visual Studio 2015.

If you want re-generate solution file or generate solution file for Visual Studio 2015,

    > premake5 vs2015

For Visual Studio 2017,

    > premake5 vs2017


## Usage

Edit `config.json`, then

    $ ./bin/native/Release/view

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

