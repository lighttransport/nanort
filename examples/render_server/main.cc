/* Copyright (c) 2013-2017 the Civetweb developers
 * Copyright (c) 2013 No Face Press, LLC
 * License http://opensource.org/licenses/mit-license.php MIT License
 */

// Simple example program on how to use Embedded C++ interface.

#include "CivetServer.h"
#include "json.hpp"

#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <atomic>  // C++11
#include <chrono>  // C++11
#include <mutex> // C++11
#include <thread> // C++11

#include "render.h"
#include "trackball.h"

#define DOCUMENT_ROOT "."
#define PORT "8081"
#define EXIT_URI "/exit"
bool exitNow = false;
using json = nlohmann::json;

example::Renderer gRenderer;

std::atomic<bool> gRenderQuit;
std::atomic<bool> gRenderRefresh;
std::atomic<bool> gRenderCancel;
json gRenderConfig;
std::mutex gMutex;

std::vector<float> gRGBA;
std::vector<int> gSampleCounts;
float gCurrQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};


void InitRender(json& scene)
{
    scene["pass"] = 0;
    scene["maxPasses"] = 10;
    int width = scene["resolutionX"];
    int height = scene["resolutionY"];
    gRGBA.resize(width * height * 4);
    std::fill(gRGBA.begin(), gRGBA.end(), 0.0);
    gSampleCounts.resize(width * height);
    std::fill(gSampleCounts.begin(), gSampleCounts.end(), 0.0);
    trackball(gCurrQuat, 0.0f, 0.0f, 0.0f, 0.0f);
}

void RequestRender() {
    {
        std::lock_guard<std::mutex> guard(gMutex);
        gRenderConfig["pass"] = 0;
    }

    gRenderRefresh = true;
    gRenderCancel = true;
}

class ExitHandler : public CivetHandler
{
  public:
	bool
	handleGet(CivetServer *server, struct mg_connection *conn)
	{
		(void)server;

		mg_printf(conn,
		          "HTTP/1.1 200 OK\r\nContent-Type: "
		          "text/plain\r\nConnection: close\r\n\r\n");
		mg_printf(conn, "Bye!\n");
		exitNow = true;
		return true;
	}
};

class FooHandler : public CivetHandler
{
  public:
	bool
	handleGet(CivetServer *server, struct mg_connection *conn)
	{
		(void)server;

		/* Handler may access the request info using mg_get_request_info */
		const struct mg_request_info *req_info = mg_get_request_info(conn);

		mg_printf(conn,
		          "HTTP/1.1 200 OK\r\nContent-Type: "
		          "text/html\r\nConnection: close\r\n\r\n");

		mg_printf(conn, "<html><body>\n");
		mg_printf(conn, "<h2>This is the Foo GET handler!!!</h2>\n");
		mg_printf(conn,
		          "<p>The request was:<br><pre>%s %s HTTP/%s</pre></p>\n",
		          req_info->request_method,
		          req_info->request_uri,
		          req_info->http_version);
		mg_printf(conn, "</body></html>\n");

		return true;
	}
	bool
	handlePost(CivetServer *server, struct mg_connection *conn)
	{
		(void)server;

		/* Handler may access the request info using mg_get_request_info */
		const struct mg_request_info *req_info = mg_get_request_info(conn);
		long long rlen, wlen;
		long long nlen = 0;
		long long tlen = req_info->content_length;
		char buf[1024];

		mg_printf(conn,
		          "HTTP/1.1 200 OK\r\nContent-Type: "
		          "text/html\r\nConnection: close\r\n\r\n");

		mg_printf(conn, "<html><body>\n");
		mg_printf(conn, "<h2>This is the Foo POST handler!!!</h2>\n");
		mg_printf(conn,
		          "<p>The request was:<br><pre>%s %s HTTP/%s</pre></p>\n",
		          req_info->request_method,
		          req_info->request_uri,
		          req_info->http_version);
		mg_printf(conn, "<p>Content Length: %li</p>\n", (long)tlen);
		mg_printf(conn, "<pre>\n");

		while (nlen < tlen) {
			rlen = tlen - nlen;
			if (rlen > sizeof(buf)) {
				rlen = sizeof(buf);
			}
			rlen = mg_read(conn, buf, (size_t)rlen);
			if (rlen <= 0) {
				break;
			}
			wlen = mg_write(conn, buf, (size_t)rlen);
			if (wlen != rlen) {
				break;
			}
			nlen += wlen;
		}

		mg_printf(conn, "\n</pre>\n");
		mg_printf(conn, "</body></html>\n");

		return true;
	}
};

class WebSocketHandler : public CivetWebSocketHandler {

	virtual bool handleConnection(CivetServer *server,
	                              const struct mg_connection *conn) {
		printf("WS connected\n");
		return true;
	}

	virtual void handleReadyState(CivetServer *server,
	                              struct mg_connection *conn) {
		printf("WS ready\n");

		const char *text = "Hello from the websocket ready handler";
		mg_websocket_write(conn, WEBSOCKET_OPCODE_TEXT, text, strlen(text));
	}

	virtual bool handleData(CivetServer *server,
	                        struct mg_connection *conn,
	                        int bits,
	                        char *data,
	                        size_t data_len) {
		printf("WS got %lu bytes: \n", (long unsigned)data_len);
		fwrite(data, 1, data_len, stdout);
		printf("\n");

        json sceneData;
        try {
            sceneData = json::parse(std::string(data, data + data_len));
        } catch (std::exception e) {
            std::cerr << "json parse error\n"  << e.what() << std::endl;
            return true;
        }

        gRenderConfig = sceneData["params"];
        printf("InitRender\n");
        InitRender(gRenderConfig);
		printf("Rendering start\n");
        for(int i = 0 ; i < gRenderConfig["maxPasses"] ; i++){
            bool ret = gRenderer.Render(&gRGBA.at(0), &gSampleCounts.at(0),
                                        gCurrQuat, gRenderConfig, gRenderCancel);
            printf("passed %d\n", i);
            gRenderConfig["pass"] = (int)gRenderConfig["pass"] + 1;
        }
        const int width = gRenderConfig["resolutionX"];
        const int height = gRenderConfig["resolutionY"];
        const int n = 3;
        std::vector<float> buf(width * height * n);
        for (size_t i = 0; i < gRGBA.size() / 4; i++) {
            buf[n * i + 0] = gRGBA[4 * i + 0];
            buf[n * i + 1] = gRGBA[4 * i + 1];
            buf[n * i + 2] = gRGBA[4 * i + 2];
//                buf[4 * i + 3] = gRGBA[4 * i + 3];
            if (gSampleCounts[i] > 0) {
                buf[n * i + 0] /= static_cast<float>(gSampleCounts[i]);
                buf[n * i + 1] /= static_cast<float>(gSampleCounts[i]);
                buf[n * i + 2] /= static_cast<float>(gSampleCounts[i]);
//                    buf[4 * i + 3] /= static_cast<float>(gSampleCounts[i]);
            }
        }
		printf("Send %dx%d RGB Image \n", width, height);
        mg_websocket_write(conn, WEBSOCKET_OPCODE_BINARY,
                           reinterpret_cast<const char*>(&buf.at(0)),
                           sizeof(float) * width * height * n);

		return true;
	}

	virtual void handleClose(CivetServer *server,
	                         const struct mg_connection *conn) {
		printf("WS closed\n");
	}
};

int
main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;

	const char *options[] = {
	    "document_root", DOCUMENT_ROOT, "listening_ports", PORT, 0};

    std::vector<std::string> cpp_options;
    for (int i=0; i<(sizeof(options)/sizeof(options[0])-1); i++) {
        cpp_options.push_back(options[i]);
    }

	// CivetServer server(options); // <-- C style start
	CivetServer server(cpp_options); // <-- C++ style start

	ExitHandler h_exit;
	server.addHandler(EXIT_URI, h_exit);

	WebSocketHandler h_websocket;
	server.addWebSocketHandler("/websocket", h_websocket);
	printf("Run websocket example at http://localhost:%s/websocket\n", PORT);

    std::string obj_filename = "scene.obj";
    float scene_scale = 1.0;
    bool obj_ret = gRenderer.LoadObjMesh(obj_filename.c_str(),
                                         scene_scale);
    if (!obj_ret) {
        fprintf(stderr, "Failed to load [ %s ]\n",
                obj_filename.c_str());
        return -1;
    }
    gRenderer.BuildBVH();

	while (!exitNow) {
#ifdef _WIN32
		Sleep(1000);
#else
		sleep(1);
#endif
	}

	printf("Bye!\n");

	return 0;
}
