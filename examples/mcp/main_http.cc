#include "nanort_mcp.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }
    
    std::cout << "Starting NanoRT MCP Server (HTTP transport) on port " << port << "..." << std::endl;
    
    std::unique_ptr<mcp::HttpTransport> http_transport(new mcp::HttpTransport(port));
    if (!http_transport->start()) {
        std::cerr << "Failed to start HTTP server on port " << port << std::endl;
        return 1;
    }
    
    std::unique_ptr<mcp::Transport> transport(http_transport.release());
    nanort_mcp::NanoRTMcpServer server(std::move(transport));
    
    std::cout << "Server ready, waiting for HTTP requests..." << std::endl;
    server.run();
    
    return 0;
}