#include "nanort_mcp.h"
#include <iostream>
#include <memory>

int main() {
    std::cerr << "Starting NanoRT MCP Server (stdio transport)..." << std::endl;
    
    std::unique_ptr<mcp::Transport> transport(new mcp::StdioTransport());
    nanort_mcp::NanoRTMcpServer server(std::move(transport));
    
    std::cerr << "Server ready, waiting for MCP messages..." << std::endl;
    server.run();
    
    return 0;
}