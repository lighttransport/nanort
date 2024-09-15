# Simple GuassianSplatting rendering example with GUI support(bullet3's OpenGLWindow + ImGui).

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* premake5
* OpenGL 2.x

## TODO

* [ ] Color

## Build 

    $ premake5 gmake
    $ make

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/gsplat

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

