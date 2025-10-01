#ifndef MCP_SERVER_H
#define MCP_SERVER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace mcp {

struct JsonValue {
    enum Type { STRING, NUMBER, BOOLEAN, OBJECT, ARRAY, NULL_VALUE };
    Type type;
    std::string string_value;
    double number_value;
    bool boolean_value;
    std::map<std::string, std::shared_ptr<JsonValue>> object_value;
    std::vector<std::shared_ptr<JsonValue>> array_value;
    
    JsonValue(Type t = NULL_VALUE) : type(t), number_value(0.0), boolean_value(false) {}
    static std::shared_ptr<JsonValue> fromString(const std::string& str);
    std::string toString() const;
};

struct JsonRpcRequest {
    std::string jsonrpc;
    std::string method;
    std::shared_ptr<JsonValue> params;
    std::shared_ptr<JsonValue> id;
    
    static JsonRpcRequest fromJson(const std::string& json);
};

struct JsonRpcResponse {
    std::string jsonrpc;
    std::shared_ptr<JsonValue> result;
    std::shared_ptr<JsonValue> error;
    std::shared_ptr<JsonValue> id;
    
    std::string toJson() const;
};

struct ToolParameter {
    std::string name;
    std::string type;
    std::string description;
    bool required;
    
    ToolParameter(const std::string& n, const std::string& t, const std::string& d, bool r = false)
        : name(n), type(t), description(d), required(r) {}
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::vector<ToolParameter> parameters;
    
    ToolDefinition(const std::string& n, const std::string& d) : name(n), description(d) {}
};

class Transport {
public:
    virtual ~Transport() = default;
    virtual void send(const std::string& message) = 0;
    virtual std::string receive() = 0;
    virtual bool isConnected() const = 0;
};

class StdioTransport : public Transport {
public:
    void send(const std::string& message) override;
    std::string receive() override;
    bool isConnected() const override { return true; }
};

class HttpTransport : public Transport {
private:
    int server_socket;
    int client_socket;
    int port;
    bool running;
    
public:
    HttpTransport(int port = 8080);
    ~HttpTransport();
    
    bool start();
    void stop();
    void send(const std::string& message) override;
    std::string receive() override;
    bool isConnected() const override;
    
private:
    void handleConnection();
    std::string parseHttpRequest(const std::string& request);
    std::string createHttpResponse(const std::string& content);
};

class McpServer {
private:
    std::unique_ptr<Transport> transport;
    std::vector<ToolDefinition> tools;
    std::map<std::string, std::function<JsonRpcResponse(const JsonRpcRequest&)>> tool_handlers;
    bool initialized;
    
public:
    McpServer(std::unique_ptr<Transport> t);
    ~McpServer() = default;
    
    void addTool(const ToolDefinition& tool, 
                 std::function<JsonRpcResponse(const JsonRpcRequest&)> handler);
    
    void run();
    void stop();
    
private:
    JsonRpcResponse handleRequest(const JsonRpcRequest& request);
    JsonRpcResponse handleInitialize(const JsonRpcRequest& request);
    JsonRpcResponse handleListTools(const JsonRpcRequest& request);
    JsonRpcResponse handleCallTool(const JsonRpcRequest& request);
    
    JsonRpcResponse createErrorResponse(const JsonRpcRequest& request, 
                                      int code, const std::string& message);
};

} // namespace mcp

#endif // MCP_SERVER_H