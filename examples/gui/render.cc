#include "render.h"

#include <chrono>  // C++11
#include <thread>  // C++11
#include <vector>  // C++11

namespace example {

bool Render(float* rgba, float *aux_rgba, const RenderConfig& config, std::atomic<bool>& cancelFlag)
{
  auto kCancelFlagCheckMilliSeconds = 300;

  std::vector<std::thread> workers;
  std::atomic<int> i( 0 );

  uint32_t num_threads = std::max( 1U, std::thread::hardware_concurrency() );

  auto startT = std::chrono::system_clock::now();

  for( auto t = 0; t < num_threads; t++ )
  {
    workers.push_back( std::thread( [&, t]() {

      int y = 0;
      while ((y = i++) < config.height) {
        auto currT = std::chrono::system_clock::now();

        std::chrono::duration<double, std::milli> ms = currT - startT;
        // Check cancel flag
        if (ms.count() > kCancelFlagCheckMilliSeconds) {
          if (cancelFlag) {
            break;
          }
        }

        // draw dash line to aux buffer for progress.
        for (int x = 0; x < config.width; x++) {
          float c = (x / 8) % 2;
          aux_rgba[4*(y*config.width+x)+0] = c;
          aux_rgba[4*(y*config.width+x)+1] = c;
          aux_rgba[4*(y*config.width+x)+2] = c;
          aux_rgba[4*(y*config.width+x)+3] = 0.0f;
        }

        //std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );

        for (int x = 0; x < config.width; x++) {
          rgba[4*(y*config.width+x)+0] = x / static_cast<float>(config.width);
          rgba[4*(y*config.width+x)+1] = y / static_cast<float>(config.height);
          rgba[4*(y*config.width+x)+2] = config.pass / static_cast<float>(config.max_passes);
          rgba[4*(y*config.width+x)+3] = 1.0f;
        }

        for (int x = 0; x < config.width; x++) {
          aux_rgba[4*(y*config.width+x)+0] = 0.0f;
          aux_rgba[4*(y*config.width+x)+1] = 0.0f;
          aux_rgba[4*(y*config.width+x)+2] = 0.0f;
          aux_rgba[4*(y*config.width+x)+3] = 0.0f;
        }
      }
    }));
  }

  for (auto &t : workers) {
    t.join();
  }

  return true;
  
};

}  // namespace example
