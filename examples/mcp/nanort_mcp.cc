#include "nanort_mcp.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace nanort_mcp {

NanoRTMcpServer::NanoRTMcpServer(std::unique_ptr<mcp::Transport> transport) 
    : geometry_loaded(false) {
    
    server.reset(new mcp::McpServer(std::move(transport)));
    
    mcp::ToolDefinition load_tool("load", "Load 3D geometry from a file");
    load_tool.parameters.push_back(mcp::ToolParameter("filename", "string", "Path to the geometry file", true));
    
    mcp::ToolDefinition render_tool("render", "Render the loaded geometry and return AOV floating point buffer");
    render_tool.parameters.push_back(mcp::ToolParameter("width", "number", "Image width", false));
    render_tool.parameters.push_back(mcp::ToolParameter("height", "number", "Image height", false));
    render_tool.parameters.push_back(mcp::ToolParameter("samples", "number", "Number of samples per pixel", false));
    render_tool.parameters.push_back(mcp::ToolParameter("camera_pos", "array", "Camera position [x,y,z]", false));
    render_tool.parameters.push_back(mcp::ToolParameter("camera_target", "array", "Camera target [x,y,z]", false));
    render_tool.parameters.push_back(mcp::ToolParameter("fov", "number", "Field of view in degrees", false));
    
    server->addTool(load_tool, [this](const mcp::JsonRpcRequest& req) { return handleLoadTool(req); });
    server->addTool(render_tool, [this](const mcp::JsonRpcRequest& req) { return handleRenderTool(req); });
}

void NanoRTMcpServer::run() {
    server->run();
}

mcp::JsonRpcResponse NanoRTMcpServer::handleLoadTool(const mcp::JsonRpcRequest& request) {
    mcp::JsonRpcResponse response;
    response.jsonrpc = "2.0";
    response.id = request.id;
    
    std::string filename = "teapot.obj";
    
    if (!loadObjFile(filename)) {
        auto error = std::make_shared<mcp::JsonValue>(mcp::JsonValue::OBJECT);
        auto code_val = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
        code_val->number_value = -1;
        error->object_value["code"] = code_val;
        error->object_value["message"] = mcp::JsonValue::fromString("Failed to load geometry file: " + filename);
        response.error = error;
        return response;
    }
    
    auto result = std::make_shared<mcp::JsonValue>(mcp::JsonValue::OBJECT);
    result->object_value["success"] = std::make_shared<mcp::JsonValue>(mcp::JsonValue::BOOLEAN);
    result->object_value["success"]->boolean_value = true;
    result->object_value["message"] = mcp::JsonValue::fromString("Geometry loaded successfully");
    
    auto vertices_count = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
    vertices_count->number_value = vertices.size() / 3;
    result->object_value["vertices_count"] = vertices_count;
    
    auto triangles_count = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
    triangles_count->number_value = faces.size() / 3;
    result->object_value["triangles_count"] = triangles_count;
    
    response.result = result;
    return response;
}

mcp::JsonRpcResponse NanoRTMcpServer::handleRenderTool(const mcp::JsonRpcRequest& request) {
    mcp::JsonRpcResponse response;
    response.jsonrpc = "2.0";
    response.id = request.id;
    
    if (!geometry_loaded) {
        auto error = std::make_shared<mcp::JsonValue>(mcp::JsonValue::OBJECT);
        auto code_val = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
        code_val->number_value = -2;
        error->object_value["code"] = code_val;
        error->object_value["message"] = mcp::JsonValue::fromString("No geometry loaded. Use 'load' tool first.");
        response.error = error;
        return response;
    }
    
    std::vector<float> buffer = renderImage();
    
    auto result = std::make_shared<mcp::JsonValue>(mcp::JsonValue::OBJECT);
    result->object_value["width"] = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
    result->object_value["width"]->number_value = render_settings.width;
    
    result->object_value["height"] = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
    result->object_value["height"]->number_value = render_settings.height;
    
    result->object_value["channels"] = std::make_shared<mcp::JsonValue>(mcp::JsonValue::NUMBER);
    result->object_value["channels"]->number_value = 4; // RGBA
    
    result->object_value["format"] = mcp::JsonValue::fromString("float32");
    
    std::string encoded_buffer = encodeFloatBuffer(buffer, render_settings.width, render_settings.height, 4);
    result->object_value["data"] = mcp::JsonValue::fromString(encoded_buffer);
    
    response.result = result;
    return response;
}

bool NanoRTMcpServer::loadObjFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    
    vertices.clear();
    faces.clear();
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        
        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        } else if (prefix == "f") {
            std::string v1_str, v2_str, v3_str;
            iss >> v1_str >> v2_str >> v3_str;
            
            unsigned int v1 = std::stoi(v1_str.substr(0, v1_str.find('/'))) - 1;
            unsigned int v2 = std::stoi(v2_str.substr(0, v2_str.find('/'))) - 1;
            unsigned int v3 = std::stoi(v3_str.substr(0, v3_str.find('/'))) - 1;
            
            faces.push_back(v1);
            faces.push_back(v2);
            faces.push_back(v3);
        }
    }
    
    if (vertices.empty() || faces.empty()) {
        std::cerr << "No valid geometry found in file" << std::endl;
        return false;
    }
    
    nanort::TriangleMesh<float> triangle_mesh(&vertices[0], &faces[0], sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(&vertices[0], &faces[0], sizeof(float) * 3);
    
    nanort::BVHBuildOptions<float> build_options;
    build_options.cache_bbox = false;
    
    bool ret = accel.Build(static_cast<unsigned int>(faces.size() / 3), triangle_mesh, triangle_pred, build_options);
    if (!ret) {
        std::cerr << "Failed to build BVH" << std::endl;
        return false;
    }
    
    geometry_loaded = true;
    std::cout << "Loaded " << vertices.size() / 3 << " vertices and " << faces.size() / 3 << " triangles" << std::endl;
    return true;
}

std::vector<float> NanoRTMcpServer::renderImage() {
    int width = render_settings.width;
    int height = render_settings.height;
    std::vector<float> buffer(width * height * 4, 0.0f); // RGBA
    
    float aspect = float(width) / float(height);
    float fov_rad = render_settings.fov * M_PI / 180.0f;
    float tan_half_fov = std::tan(fov_rad * 0.5f);
    
    // Calculate camera vectors
    float view_dir[3] = {
        render_settings.camera_target[0] - render_settings.camera_pos[0],
        render_settings.camera_target[1] - render_settings.camera_pos[1],
        render_settings.camera_target[2] - render_settings.camera_pos[2]
    };
    
    // Normalize view direction
    float len = std::sqrt(view_dir[0] * view_dir[0] + view_dir[1] * view_dir[1] + view_dir[2] * view_dir[2]);
    view_dir[0] /= len; view_dir[1] /= len; view_dir[2] /= len;
    
    // Calculate right vector
    float right[3];
    right[0] = view_dir[1] * render_settings.camera_up[2] - view_dir[2] * render_settings.camera_up[1];
    right[1] = view_dir[2] * render_settings.camera_up[0] - view_dir[0] * render_settings.camera_up[2];
    right[2] = view_dir[0] * render_settings.camera_up[1] - view_dir[1] * render_settings.camera_up[0];
    
    len = std::sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
    right[0] /= len; right[1] /= len; right[2] /= len;
    
    // Calculate up vector
    float up[3];
    up[0] = right[1] * view_dir[2] - right[2] * view_dir[1];
    up[1] = right[2] * view_dir[0] - right[0] * view_dir[2];
    up[2] = right[0] * view_dir[1] - right[1] * view_dir[0];
    
    nanort::BVHTraceOptions trace_options;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float u = (2.0f * float(x) / float(width) - 1.0f) * tan_half_fov * aspect;
            float v = (1.0f - 2.0f * float(y) / float(height)) * tan_half_fov;
            
            float ray_dir[3] = {
                view_dir[0] + u * right[0] + v * up[0],
                view_dir[1] + u * right[1] + v * up[1],
                view_dir[2] + u * right[2] + v * up[2]
            };
            
            len = std::sqrt(ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2]);
            ray_dir[0] /= len; ray_dir[1] /= len; ray_dir[2] /= len;
            
            nanort::Ray<float> ray;
            ray.org[0] = render_settings.camera_pos[0];
            ray.org[1] = render_settings.camera_pos[1];
            ray.org[2] = render_settings.camera_pos[2];
            ray.dir[0] = ray_dir[0];
            ray.dir[1] = ray_dir[1];
            ray.dir[2] = ray_dir[2];
            
            nanort::TriangleIntersector<> triangle_intersector(&vertices[0], &faces[0], sizeof(float) * 3);
            nanort::TriangleIntersection<> isect;
            bool hit = accel.Traverse(ray, triangle_intersector, &isect, trace_options);
            
            int idx = (y * width + x) * 4;
            if (hit) {
                // Simple shading based on depth
                float depth = isect.t / 10.0f; // normalize depth
                depth = std::min(1.0f, std::max(0.0f, depth));
                
                buffer[idx + 0] = 1.0f - depth; // R
                buffer[idx + 1] = 1.0f - depth; // G
                buffer[idx + 2] = 1.0f - depth; // B
                buffer[idx + 3] = 1.0f;         // A
            } else {
                buffer[idx + 0] = 0.0f; // R
                buffer[idx + 1] = 0.0f; // G
                buffer[idx + 2] = 0.1f; // B (background)
                buffer[idx + 3] = 1.0f; // A
            }
        }
    }
    
    return buffer;
}

std::string NanoRTMcpServer::encodeFloatBuffer(const std::vector<float>& buffer, int width, int height, int channels) {
    std::ostringstream oss;
    oss << "data:application/octet-stream;base64,";
    
    // Simple base64-like encoding for demonstration
    // In a real implementation, you'd use proper base64 encoding
    for (size_t i = 0; i < buffer.size(); i += 4) {
        if (i + 3 < buffer.size()) {
            oss << buffer[i] << "," << buffer[i+1] << "," << buffer[i+2] << "," << buffer[i+3];
            if (i + 4 < buffer.size()) oss << ";";
        }
    }
    
    return oss.str();
}

} // namespace nanort_mcp