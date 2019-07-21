# Simple GUI example with ptex texture mapping.

## Coordinates

Right-handed coorinate, Y up, counter clock-wise normal definition.

## Requirements

* git
* cmake
* OpenGL 2.x

## Setup

Clone `ptex-aarch64`(Added reading ptex data from memory) https://github.com/syoyo/ptex-aarch64.git to this directory.

```
$ cd $nanort/examples/ptex
$ git clone https://github.com/syoyo/ptex-aarch64.git
```

### cmake build

#### Build on Linux/MacOSX

```
$ rmdir build
$ mkdir build
$ cd build
$ cmake ..
$ make
```

#### Build on Windows

```
$ rmdir /s /q build
$ mkdir build
$ cd build
$ cmake -G "Visual Studio 15 2017" -A x64 ..
```

Open `.sln` and build it with Visual Studio 2017.

## Usage

Download some ptex scene for example from http://ptex.us/samples.html

Edit `config.json`, then

```
$ cd build
$ ./ptexrt
```

### Mouse operation

* left mouse = rotate
* shift + left mouse = translate
* tab + left mouse = dolly(Z axis)

