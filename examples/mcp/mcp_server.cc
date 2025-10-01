#include "mcp_server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#define SOCKET_ERROR -1
#define INVALID_SOCKET -1
#define closesocket close
#endif

namespace mcp {

std::shared_ptr<JsonValue> JsonValue::fromString(const std::string& str) {
    auto value = std::make_shared<JsonValue>(STRING);
    value->string_value = str;
    return value;
}

std::string JsonValue::toString() const {
    switch (type) {
        case STRING:
            return "\"" + string_value + "\"";
        case NUMBER:
            return std::to_string(number_value);
        case BOOLEAN:
            return boolean_value ? "true" : "false";
        case NULL_VALUE:
            return "null";
        case OBJECT: {
            std::string result = "{";
            bool first = true;
            for (const auto& pair : object_value) {
                if (!first) result += ",";
                result += "\"" + pair.first + "\":" + pair.second->toString();
                first = false;
            }
            result += "}";
            return result;
        }
        case ARRAY: {
            std::string result = "[";
            bool first = true;
            for (const auto& item : array_value) {
                if (!first) result += ",";
                result += item->toString();
                first = false;
            }
            result += "]";
            return result;
        }
    }
    return "null";
}

JsonRpcRequest JsonRpcRequest::fromJson(const std::string& json) {
    JsonRpcRequest request;
    
    size_t jsonrpc_pos = json.find("\"jsonrpc\"");
    if (jsonrpc_pos != std::string::npos) {
        size_t start = json.find("\"", jsonrpc_pos + 9);
        size_t end = json.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            request.jsonrpc = json.substr(start + 1, end - start - 1);
        }
    }
    
    size_t method_pos = json.find("\"method\"");
    if (method_pos != std::string::npos) {
        size_t start = json.find("\"", method_pos + 8);
        size_t end = json.find("\"", start + 1);
        if (start != std::string::npos && end != std::string::npos) {
            request.method = json.substr(start + 1, end - start - 1);
        }
    }
    
    size_t params_pos = json.find("\"params\"");
    if (params_pos != std::string::npos) {
        size_t colon = json.find(":", params_pos);
        size_t start = colon + 1;
        while (start < json.length() && (json[start] == ' ' || json[start] == '\t')) start++;
        
        if (json[start] == '{') {
            // Find matching closing brace
            int brace_count = 0;
            size_t end = start;
            do {
                if (json[end] == '{') brace_count++;
                if (json[end] == '}') brace_count--;
                end++;
            } while (end < json.length() && brace_count > 0);
            
            if (brace_count == 0) {
                std::string params_json = json.substr(start, end - start);
                auto params_val = std::make_shared<JsonValue>(JsonValue::STRING);
                params_val->string_value = params_json; // Store as string for simple parsing
                request.params = params_val;
            }
        }
    }
    
    size_t id_pos = json.find("\"id\"");
    if (id_pos != std::string::npos) {
        size_t colon = json.find(":", id_pos);
        size_t start = colon + 1;
        while (start < json.length() && (json[start] == ' ' || json[start] == '\t')) start++;
        
        if (json[start] == '"') {
            size_t end = json.find("\"", start + 1);
            if (end != std::string::npos) {
                auto id_val = std::make_shared<JsonValue>(JsonValue::STRING);
                id_val->string_value = json.substr(start + 1, end - start - 1);
                request.id = id_val;
            }
        } else if (isdigit(json[start]) || json[start] == '-') {
            size_t end = start;
            while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-')) end++;
            auto id_val = std::make_shared<JsonValue>(JsonValue::NUMBER);
            id_val->number_value = std::stod(json.substr(start, end - start));
            request.id = id_val;
        }
    }
    
    return request;
}

std::string JsonRpcResponse::toJson() const {
    std::string json = "{\"jsonrpc\":\"2.0\"";
    
    if (id) {
        json += ",\"id\":" + id->toString();
    }
    
    if (result) {
        json += ",\"result\":" + result->toString();
    }
    
    if (error) {
        json += ",\"error\":" + error->toString();
    }
    
    json += "}";
    return json;
}

void StdioTransport::send(const std::string& message) {
    std::cout << message << std::endl;
    std::cout.flush();
}

std::string StdioTransport::receive() {
    std::string line;
    if (std::getline(std::cin, line)) {
        return line;
    }
    return "";
}

#ifdef _WIN32
class WinsockInit {
public:
    WinsockInit() {
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2,2), &wsaData);
    }
    ~WinsockInit() {
        WSACleanup();
    }
};
static WinsockInit winsock_init;
#endif

HttpTransport::HttpTransport(int port) : server_socket(INVALID_SOCKET), client_socket(INVALID_SOCKET), port(port), running(false) {
}

HttpTransport::~HttpTransport() {
    stop();
}

bool HttpTransport::start() {
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == INVALID_SOCKET) {
        return false;
    }
    
    int opt = 1;
#ifdef _WIN32
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
#else
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);
    
    if (bind(server_socket, (struct sockaddr*)&address, sizeof(address)) < 0) {
        closesocket(server_socket);
        return false;
    }
    
    if (listen(server_socket, 3) < 0) {
        closesocket(server_socket);
        return false;
    }
    
    running = true;
    std::cout << "HTTP MCP Server listening on port " << port << std::endl;
    return true;
}

void HttpTransport::stop() {
    running = false;
    if (client_socket != INVALID_SOCKET) {
        closesocket(client_socket);
        client_socket = INVALID_SOCKET;
    }
    if (server_socket != INVALID_SOCKET) {
        closesocket(server_socket);
        server_socket = INVALID_SOCKET;
    }
}

void HttpTransport::send(const std::string& message) {
    if (client_socket != INVALID_SOCKET) {
        std::string response = createHttpResponse(message);
        ::send(client_socket, response.c_str(), response.length(), 0);
    }
}

std::string HttpTransport::receive() {
    if (!running) return "";
    
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    
    client_socket = accept(server_socket, (struct sockaddr*)&address, (socklen_t*)&addrlen);
    if (client_socket == INVALID_SOCKET) {
        return "";
    }
    
    char buffer[4096] = {0};
    int bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_read <= 0) {
        closesocket(client_socket);
        client_socket = INVALID_SOCKET;
        return "";
    }
    
    std::string request(buffer, bytes_read);
    return parseHttpRequest(request);
}

bool HttpTransport::isConnected() const {
    return running && server_socket != INVALID_SOCKET;
}

std::string HttpTransport::parseHttpRequest(const std::string& request) {
    size_t body_start = request.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        return request.substr(body_start + 4);
    }
    return "";
}

std::string HttpTransport::createHttpResponse(const std::string& content) {
    std::stringstream response;
    response << "HTTP/1.1 200 OK\r\n";
    response << "Content-Type: application/json\r\n";
    response << "Content-Length: " << content.length() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: POST, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    response << "\r\n";
    response << content;
    return response.str();
}

McpServer::McpServer(std::unique_ptr<Transport> t) : transport(std::move(t)), initialized(false) {
}

void McpServer::addTool(const ToolDefinition& tool, 
                       std::function<JsonRpcResponse(const JsonRpcRequest&)> handler) {
    tools.push_back(tool);
    tool_handlers[tool.name] = handler;
}

void McpServer::run() {
    while (transport->isConnected()) {
        std::string message = transport->receive();
        if (message.empty()) continue;
        
        std::cerr << "Received: " << message << std::endl;
        
        JsonRpcRequest request = JsonRpcRequest::fromJson(message);
        JsonRpcResponse response = handleRequest(request);
        
        std::string response_json = response.toJson();
        std::cerr << "Sending: " << response_json << std::endl;
        transport->send(response_json);
    }
}

void McpServer::stop() {
    // Transport should handle cleanup
}

JsonRpcResponse McpServer::handleRequest(const JsonRpcRequest& request) {
    if (request.method == "initialize") {
        return handleInitialize(request);
    } else if (request.method == "tools/list") {
        return handleListTools(request);
    } else if (request.method == "tools/call") {
        return handleCallTool(request);
    } else {
        return createErrorResponse(request, -32601, "Method not found");
    }
}

JsonRpcResponse McpServer::handleInitialize(const JsonRpcRequest& request) {
    initialized = true;
    
    JsonRpcResponse response;
    response.jsonrpc = "2.0";
    response.id = request.id;
    
    auto result = std::make_shared<JsonValue>(JsonValue::OBJECT);
    result->object_value["protocolVersion"] = JsonValue::fromString("2024-11-05");
    
    auto capabilities = std::make_shared<JsonValue>(JsonValue::OBJECT);
    auto tools_cap = std::make_shared<JsonValue>(JsonValue::OBJECT);
    capabilities->object_value["tools"] = tools_cap;
    result->object_value["capabilities"] = capabilities;
    
    auto server_info = std::make_shared<JsonValue>(JsonValue::OBJECT);
    server_info->object_value["name"] = JsonValue::fromString("nanort-mcp-server");
    server_info->object_value["version"] = JsonValue::fromString("1.0.0");
    result->object_value["serverInfo"] = server_info;
    
    response.result = result;
    return response;
}

JsonRpcResponse McpServer::handleListTools(const JsonRpcRequest& request) {
    JsonRpcResponse response;
    response.jsonrpc = "2.0";
    response.id = request.id;
    
    auto result = std::make_shared<JsonValue>(JsonValue::OBJECT);
    auto tools_array = std::make_shared<JsonValue>(JsonValue::ARRAY);
    
    for (const auto& tool : tools) {
        auto tool_obj = std::make_shared<JsonValue>(JsonValue::OBJECT);
        tool_obj->object_value["name"] = JsonValue::fromString(tool.name);
        tool_obj->object_value["description"] = JsonValue::fromString(tool.description);
        
        auto input_schema = std::make_shared<JsonValue>(JsonValue::OBJECT);
        input_schema->object_value["type"] = JsonValue::fromString("object");
        
        auto properties = std::make_shared<JsonValue>(JsonValue::OBJECT);
        auto required_array = std::make_shared<JsonValue>(JsonValue::ARRAY);
        
        for (const auto& param : tool.parameters) {
            auto param_obj = std::make_shared<JsonValue>(JsonValue::OBJECT);
            param_obj->object_value["type"] = JsonValue::fromString(param.type);
            param_obj->object_value["description"] = JsonValue::fromString(param.description);
            properties->object_value[param.name] = param_obj;
            
            if (param.required) {
                required_array->array_value.push_back(JsonValue::fromString(param.name));
            }
        }
        
        input_schema->object_value["properties"] = properties;
        input_schema->object_value["required"] = required_array;
        tool_obj->object_value["inputSchema"] = input_schema;
        
        tools_array->array_value.push_back(tool_obj);
    }
    
    result->object_value["tools"] = tools_array;
    response.result = result;
    return response;
}

JsonRpcResponse McpServer::handleCallTool(const JsonRpcRequest& request) {
    if (!initialized) {
        return createErrorResponse(request, -32002, "Server not initialized");
    }
    
    // Extract tool name from params
    std::string tool_name;
    if (request.params) {
        // Parse params JSON string
        std::string params_json = request.params->string_value;
        size_t name_pos = params_json.find("\"name\"");
        if (name_pos != std::string::npos) {
            size_t colon_pos = params_json.find(":", name_pos);
            if (colon_pos != std::string::npos) {
                size_t start_quote = params_json.find("\"", colon_pos);
                if (start_quote != std::string::npos) {
                    size_t end_quote = params_json.find("\"", start_quote + 1);
                    if (end_quote != std::string::npos) {
                        tool_name = params_json.substr(start_quote + 1, end_quote - start_quote - 1);
                    }
                }
            }
        }
    }
    
    auto handler = tool_handlers.find(tool_name);
    if (handler != tool_handlers.end()) {
        return handler->second(request);
    }
    
    return createErrorResponse(request, -32602, "Tool not found: " + tool_name);
}

JsonRpcResponse McpServer::createErrorResponse(const JsonRpcRequest& request, 
                                             int code, const std::string& message) {
    JsonRpcResponse response;
    response.jsonrpc = "2.0";
    response.id = request.id;
    
    auto error = std::make_shared<JsonValue>(JsonValue::OBJECT);
    auto code_val = std::make_shared<JsonValue>(JsonValue::NUMBER);
    code_val->number_value = code;
    error->object_value["code"] = code_val;
    error->object_value["message"] = JsonValue::fromString(message);
    
    response.error = error;
    return response;
}

} // namespace mcp