# Simple raytracing Minecfract model using enkiMI(W.I.P.).

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* premake5
* OpenGL 2.x

## Build

    $ premake5 gmake
    $ make

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/minecraftrender

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

## Licenses

* enkiMI : Minecraft importer. See `License.enkimi.txt` for details
* miniz : zlib license.
* r.1.0.mca : Sample MI model. see `r.1.0.mca.readme.txt` for details.

