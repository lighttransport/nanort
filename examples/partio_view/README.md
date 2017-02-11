# Simple Partio viewer using ray - sphere intersection

## Requirements

* premake5
* Partio https://www.disneyanimation.com/technology/partio.html
* OpenGL 2.x


## Build on MacOSX

    $ brew install partio

    $ premake5 gmake
    $ make

## Build on Linux

Edit the path to partio in `premake5.lua`. Then,

    $ premake5 gmake
    $ make

## Build on MinGW(Experimental)

Edit the path to partio in `premake5.lua`. Then,

    $ premake5 gmake
    $ make

## Usage

Edit `config.json`, then

    $ ./bin/native/Release/partio_view

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

