# Simple LAS(LiDAR) particle rendering example with GUI support(bullet3's OpenGLWindow + ImGui).

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* premake5
* OpenGL 2.x
* liblas
* lastools(laszip) (optional)

## TODO

* [ ] Color

## Build on Linux

Install liblas http://www.liblas.org/

Then,

    $ premake5 gmake
    $ make

## Build on MacOSX

Install liblas using brew

    $ brew install liblas
Then,

    $ premake5 gmake
    $ make

Please note that `libas` installed with brew does not support compreession(LAZ), thus if you want to use laz data, you must first decompress laz using laszip by building http://www.laszip.org/

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/lasrender

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

