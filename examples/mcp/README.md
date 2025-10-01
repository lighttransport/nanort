# NanoRT MCP (Model Context Protocol) Example

This example implements an MCP server for NanoRT ray tracing with both stdio and HTTP transports.

## Overview

The MCP server provides two tools:
- `load`: Load 3D geometry from an OBJ file
- `render`: Render the loaded geometry and return AOV floating point buffer

## Building

### Using Make
```bash
make all          # Build both versions
make stdio        # Build stdio version only
make http         # Build HTTP version only
```

### Using CMake
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Stdio Transport
```bash
./nanort_mcp_stdio
```

The stdio version communicates via standard input/output using JSON-RPC 2.0 messages.

### HTTP Transport
```bash
./nanort_mcp_http [port]
```

Default port is 8080. The HTTP version accepts JSON-RPC 2.0 messages via HTTP POST requests.

## MCP Protocol

### Initialize
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "test-client",
      "version": "1.0.0"
    }
  }
}
```

### List Tools
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
```

### Load Geometry
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "load",
    "arguments": {
      "filename": "teapot.obj"
    }
  }
}
```

### Render Scene
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "render",
    "arguments": {
      "width": 512,
      "height": 512,
      "samples": 1,
      "camera_pos": [0, 0, 5],
      "camera_target": [0, 0, 0],
      "fov": 45
    }
  }
}
```

## Implementation Details

### Transport Layer
- **StdioTransport**: Communicates via stdin/stdout for process-based integration
- **HttpTransport**: Pure C++ HTTP server using platform sockets (winsock2 on Windows, POSIX on others)

### Ray Tracing
- Uses NanoRT BVH acceleration structure
- Simple triangle intersection with Möller-Trumbore algorithm
- Basic diffuse shading based on surface normals

### AOV Buffer Format
The render tool returns a floating-point RGBA buffer encoded as a comma-separated string with metadata:
- `width`: Image width in pixels
- `height`: Image height in pixels  
- `channels`: Number of channels (4 for RGBA)
- `format`: Data format ("float32")
- `data`: Encoded pixel data

## Platform Support

- Linux: pthread support for threading
- macOS: pthread support for threading  
- Windows: Winsock2 for networking

## Dependencies

- NanoRT header-only library (../../nanort.h)
- Standard C++11 library
- Platform socket libraries (automatically linked)

## Testing

A simple cube geometry (teapot.obj) is provided for testing. The server logs to stderr while maintaining JSON-RPC communication on stdout (stdio version) or HTTP (HTTP version).