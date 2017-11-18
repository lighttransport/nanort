# Aether-render

Experimental renderer using Aether https://people.csail.mit.edu/lukea/aether/ and NanoRT.

## Status

Just finished build setup. No rendering code yet.

## Requirements

* Boost 1.64(Boost hana required. Boost 1.62 may work. Boost 1.61 doesn't work)
* Eigen3
* C++1z compiler
* CMake 3.1 or later
* clang 3.8 or later

## Supported platforms

* Ubuntu 16.04
* clang-cl.exe 4.0.1 + Visual Studio 2017.

## Setup

Clone `Aether` code to this directory.

```
$ git clone https://github.com/aekul/aether.git
```

## File layouts

```
aether-render/
.
├── aether
...

## Build(Linux)

`CMAKE_BUILD_TYPE` must be specified.

```
$ cmake -Bbuild -H.  -DCMAKE_BUILD_TYPE=Release
$ cmake --build build/
```

## Build(Windows)

Set path to clang/LLVM, Boost and Eigen in `vcbuild-clang-cl.bat`, then

```
> vcbuild-clang-cl.bat
```

Then open Visual Studio 2017 to build the code.

## Run(Linux)

### Optional

Add directory of `libc++.so` to `LD_LIBRARY_PATH`


## Licenses

* Aether : Copyright (c) 2017 aekul. MIT License.
