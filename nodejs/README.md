# NanoRT WebGPU

A Node.js implementation of NanoRT ray traversal using WebGPU compute shaders.

## Overview

This implementation provides GPU-accelerated ray traversal for NanoRT's BVH (Bounding Volume Hierarchy) structures using WebGPU compute shaders. It allows for efficient parallel ray-triangle intersection testing on the GPU.

## Features

- WebGPU compute shader-based BVH traversal
- Support for custom scenes with triangles and BVH nodes
- Batch ray processing for improved performance
- Simple camera ray generation for image rendering
- PPM image output support

## Installation

```bash
npm install
```

## Usage

### Basic Example

```javascript
const { NanoRTRenderer } = require('./index');

async function example() {
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    
    // Load a simple triangle scene
    renderer.loadSimpleScene();
    
    // Render a 256x256 image
    const image = await renderer.renderImage(256, 256);
    
    renderer.destroy();
}
```

### Custom Scene

```javascript
const { NanoRTRenderer } = require('./index');

async function customScene() {
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    
    // Define triangles
    const triangles = [{
        v0: [0, 1, 0],
        v1: [-1, -1, 0],
        v2: [1, -1, 0]
    }];
    
    // Define BVH nodes
    const bvhNodes = [{
        bmin: [-1, -1, 0],
        bmax: [1, 1, 0],
        flag: 1, // leaf node
        axis: 0,
        data: [1, 0] // 1 primitive at index 0
    }];
    
    const primitiveIndices = [0];
    
    renderer.setScene(bvhNodes, triangles, primitiveIndices);
    
    // Render rays
    const rays = [
        createRay([0, 0, -1], [0, 0, 1])
    ];
    
    const results = await renderer.renderRays(rays);
    console.log('Hit:', results[0].hit);
    
    renderer.destroy();
}
```

### Ray Definition

Rays are defined with origin, direction, and optional min/max t values:

```javascript
const { createRay } = require('./index');

const ray = createRay(
    [0, 0, -5],  // origin
    [0, 0, 1],   // direction
    0.0,         // min_t (optional)
    1000.0       // max_t (optional)
);
```

## API Reference

### NanoRTRenderer

#### Methods

- `initialize()` - Initialize WebGPU device
- `setScene(bvhNodes, triangles, primitiveIndices)` - Set custom scene
- `loadSimpleScene()` - Load a simple triangle scene for testing
- `renderRays(rays)` - Render array of rays and return intersection results
- `renderImage(width, height, camera)` - Render image from camera
- `destroy()` - Clean up resources

### Data Structures

#### Ray
```javascript
{
    origin: [x, y, z],
    direction: [x, y, z],
    min_t: number,
    max_t: number
}
```

#### Triangle
```javascript
{
    v0: [x, y, z],
    v1: [x, y, z],
    v2: [x, y, z]
}
```

#### BVH Node
```javascript
{
    bmin: [x, y, z],    // bounding box minimum
    bmax: [x, y, z],    // bounding box maximum
    flag: number,       // 0 = branch, 1 = leaf
    axis: number,       // split axis for branch nodes
    data: [number, number] // child indices or primitive count/offset
}
```

#### Intersection Result
```javascript
{
    hit: boolean,
    t: number,          // hit distance
    primitive_id: number,
    u: number,          // barycentric coordinate
    v: number           // barycentric coordinate
}
```

## Scripts

- `npm test` - Run test suite
- `npm run example` - Run example rendering

## Technical Details

### WebGPU Compute Shader

The implementation uses a WebGPU compute shader (`bvh-traversal.wgsl`) that:

1. Implements BVH traversal using a stack-based approach
2. Performs ray-AABB intersection tests for BVH nodes
3. Executes ray-triangle intersection for leaf nodes
4. Processes rays in parallel workgroups of 64

### Memory Layout

- BVH nodes: 8 floats per node (bmin, flag, bmax, axis, data0, data1)
- Triangles: 12 floats per triangle (3 vec3 + padding)
- Rays: 8 floats per ray (origin, min_t, direction, max_t)
- Results: 5 floats per result (hit, t, primitive_id, u, v)

### Performance

The implementation is optimized for batch processing of rays. For best performance:

- Process rays in batches rather than individually
- Use appropriate workgroup sizes (default: 64)
- Minimize buffer creation/destruction for repeated renders

## Requirements

- Node.js with WebGPU support
- GPU with WebGPU capabilities
- `webgpu` npm package

## Limitations

- Maximum stack depth: 64 levels
- Maximum primitives per leaf: configurable via BVH construction
- Single-precision floating point

## License

MIT License (consistent with NanoRT)