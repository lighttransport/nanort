#ifndef NANORT_MCP_H
#define NANORT_MCP_H

#include "mcp_server.h"
#include "../../nanort.h"
#include <vector>
#include <memory>

namespace nanort_mcp {

struct Vertex {
    float x, y, z;
    Vertex() : x(0), y(0), z(0) {}
    Vertex(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Triangle {
    unsigned int v0, v1, v2;
    Triangle() : v0(0), v1(0), v2(0) {}
    Triangle(unsigned int v0_, unsigned int v1_, unsigned int v2_) : v0(v0_), v1(v1_), v2(v2_) {}
};

struct RenderSettings {
    int width;
    int height;
    int samples;
    float camera_pos[3];
    float camera_target[3];
    float camera_up[3];
    float fov;
    
    RenderSettings() : width(512), height(512), samples(1), fov(45.0f) {
        camera_pos[0] = 0; camera_pos[1] = 0; camera_pos[2] = 5;
        camera_target[0] = 0; camera_target[1] = 0; camera_target[2] = 0;
        camera_up[0] = 0; camera_up[1] = 1; camera_up[2] = 0;
    }
};

// Use NanoRT's Ray type directly

// Use NanoRT's Intersection type directly


class NanoRTMcpServer {
private:
    std::unique_ptr<mcp::McpServer> server;
    std::vector<float> vertices;
    std::vector<unsigned int> faces;
    nanort::BVHAccel<float> accel;
    bool geometry_loaded;
    RenderSettings render_settings;
    
public:
    NanoRTMcpServer(std::unique_ptr<mcp::Transport> transport);
    
    void run();
    
private:
    mcp::JsonRpcResponse handleLoadTool(const mcp::JsonRpcRequest& request);
    mcp::JsonRpcResponse handleRenderTool(const mcp::JsonRpcRequest& request);
    
    bool loadObjFile(const std::string& filename);
    std::vector<float> renderImage();
    std::string encodeFloatBuffer(const std::vector<float>& buffer, int width, int height, int channels);
};

} // namespace nanort_mcp

#endif // NANORT_MCP_H