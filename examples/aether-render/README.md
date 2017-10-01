# Aether-render

Experimental Aether with NanoRT.

## Status

Just finished build setup. No rendering code yet.

## Requirements

* conan https://conan.io
  * For installing Boost 1.64 and Eigen3
* CMake 3.1 or later
* clang 3.8 or later

## Supported platforms

* Ubuntu 16.04
* Windows may work, but not tested yet.

## Setup

```
$ git clone https://github.com/aekul/aether.git
```

## File layouts

```
aether-render/
.
├── aether
``

## Build

`CMAKE_BUILD_TYPE` must be specified.

```
$ cmake -Bbuild -H.  -DCMAKE_BUILD_TYPE=Release
$ cmake --build build/
```

## Run(Linux)

### Optional

Add directory of `libc++.so` to `LD_LIBRARY_PATH`


## Licenses

* Aether : Copyright (c) 2017 aekul. MIT License.
