# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NanoRT is a single header-only modern ray tracing kernel written in C++. The core library consists of just two files: `nanort.h` (the main header-only library) and `nanort.cc` (minimal implementation file).

## Build Commands

### Quick Build and Test
```bash
# Basic compilation test with strict warnings
make

# Linting
make lint

# Development build with C++11 features
make -f Makefile.dev

# CMake build (builds examples)
mkdir build && cd build
cmake ..
make
```

### Example Building
Most examples can be built individually:
```bash
cd examples/[example_name]
make  # For examples with Makefiles
# OR
cd ../../build/examples && make  # For CMake examples
```

## Code Architecture

### Core Components

**Single Header Design**: The entire ray tracing kernel is contained in `nanort.h` (~2800 lines), making it easy to integrate into projects.

**Main Classes (nanort.h:499-820)**:
- `BVHNode`: Binary tree node for spatial partitioning
- `BVHAccel`: Main acceleration structure class that builds and traverses BVH
- `BVHBuildOptions`: Configuration for BVH construction
- `BVHTraceOptions`: Ray traversal configuration
- `BVHBuildStatistics`: Build process statistics

**Template-Based Design**: All core classes are heavily templated to support:
- Single/double precision floating point (`float`/`double`)
- Custom geometry types beyond triangles
- Custom intersection predicates

### Ray Tracing Pipeline
1. **Geometry Setup**: Prepare vertex/face arrays and custom intersectors
2. **BVH Build**: `BVHAccel::Build()` constructs spatial acceleration structure
3. **Ray Traversal**: `BVHAccel::Traverse()` finds ray intersections

### Build System Flexibility
- **CMake**: Primary build system for examples (`CMakeLists.txt`)
- **Make**: Simple builds via `Makefile` and `Makefile.dev`
- **Premake5**: Used by several examples (`examples/*/premake5.lua`)
- **Meson**: Some examples support meson builds

### Parallelization Support
- OpenMP: Enabled with `NANORT_ENABLE_PARALLEL_BUILD`
- C++11 threads: Enabled with `NANORT_USE_CPP11_FEATURE`
- CMake targets: `nanort::core`, `nanort::threads`, `nanort::openmp`

## Key Examples to Study

**Basic Usage**: `examples/objrender/` - Simple ray-triangle intersection
**Path Tracing**: `examples/path_tracer/` - Full global illumination renderer  
**Custom Geometry**: `examples/particle_primitive/`, `examples/cylinder_primitive/`
**GUI Integration**: `examples/gui/` - Interactive viewer with ImGui
**API Compatibility**: `examples/embree-api/` - Embree-compatible interface

## Development Workflow

1. **Test Core Changes**: Compile `nanort.cc` after modifying `nanort.h`
2. **Validate Examples**: Build relevant examples to test functionality
3. **Lint Code**: Run `make lint` using the included cpplint.py
4. **Performance Test**: Use `examples/benchmark/` for performance validation

## Notable Dependencies

**None for Core**: `nanort.h` is dependency-free (C++03 compatible)
**Examples Dependencies**: 
- GLM (math library)
- ImGui (GUI examples)
- STB libraries (image I/O)
- TinyObjLoader (OBJ file loading)
- Various format-specific loaders (glTF, LAS, etc.)

## Platform Support

Targets all major platforms: Windows, macOS, Linux, iOS, Android, ARM, x86, SPARC, and RISC-V. The header-only design ensures maximum portability.