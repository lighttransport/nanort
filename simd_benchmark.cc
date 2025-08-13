#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "nanort.h"

using namespace nanort;

// Benchmark configuration
const int NUM_RAYS = 100000;
const int NUM_AABBS = 1000;

void benchmark_ray_aabb_intersection() {
  std::cout << "SIMD Ray-AABB Intersection Benchmark\n";
  std::cout << "====================================\n";
  std::cout << "SIMD Path: " << NANORT_SIMD_PATH << "\n";
  std::cout << "SIMD Width: " << NANORT_SIMD_WIDTH << "\n\n";

  // Generate random test data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
  std::uniform_real_distribution<float> pos_dis(0.1f, 1.0f);
  
  // Generate rays
  std::vector<float> ray_orgs(NUM_RAYS * 3);
  std::vector<float> ray_dirs(NUM_RAYS * 3);
  std::vector<float> ray_inv_dirs(NUM_RAYS * 3);
  
  for (int i = 0; i < NUM_RAYS; i++) {
    ray_orgs[i*3 + 0] = dis(gen);
    ray_orgs[i*3 + 1] = dis(gen);
    ray_orgs[i*3 + 2] = dis(gen);
    
    ray_dirs[i*3 + 0] = dis(gen);
    ray_dirs[i*3 + 1] = dis(gen);
    ray_dirs[i*3 + 2] = dis(gen);
    
    // Normalize ray direction
    float len = std::sqrt(ray_dirs[i*3]*ray_dirs[i*3] + 
                         ray_dirs[i*3+1]*ray_dirs[i*3+1] + 
                         ray_dirs[i*3+2]*ray_dirs[i*3+2]);
    if (len > 0.0f) {
      ray_dirs[i*3 + 0] /= len;
      ray_dirs[i*3 + 1] /= len;
      ray_dirs[i*3 + 2] /= len;
    }
    
    // Compute inverse direction
    ray_inv_dirs[i*3 + 0] = (ray_dirs[i*3 + 0] != 0.0f) ? 1.0f / ray_dirs[i*3 + 0] : 1e30f;
    ray_inv_dirs[i*3 + 1] = (ray_dirs[i*3 + 1] != 0.0f) ? 1.0f / ray_dirs[i*3 + 1] : 1e30f;
    ray_inv_dirs[i*3 + 2] = (ray_dirs[i*3 + 2] != 0.0f) ? 1.0f / ray_dirs[i*3 + 2] : 1e30f;
  }
  
  // Generate AABBs
  std::vector<float> aabb_mins(NUM_AABBS * 3);
  std::vector<float> aabb_maxs(NUM_AABBS * 3);
  
  for (int i = 0; i < NUM_AABBS; i++) {
    float center_x = dis(gen);
    float center_y = dis(gen);
    float center_z = dis(gen);
    float size_x = pos_dis(gen);
    float size_y = pos_dis(gen);
    float size_z = pos_dis(gen);
    
    aabb_mins[i*3 + 0] = center_x - size_x;
    aabb_mins[i*3 + 1] = center_y - size_y;
    aabb_mins[i*3 + 2] = center_z - size_z;
    
    aabb_maxs[i*3 + 0] = center_x + size_x;
    aabb_maxs[i*3 + 1] = center_y + size_y;
    aabb_maxs[i*3 + 2] = center_z + size_z;
  }

  // Benchmark original implementation
  {
    auto start = std::chrono::high_resolution_clock::now();
    int hit_count = 0;
    
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      for (int aabb = 0; aabb < NUM_AABBS; aabb++) {
        float tmin, tmax;
        real3<float> ray_org = {ray_orgs[ray*3], ray_orgs[ray*3+1], ray_orgs[ray*3+2]};
        real3<float> ray_inv_dir = {ray_inv_dirs[ray*3], ray_inv_dirs[ray*3+1], ray_inv_dirs[ray*3+2]};
        int dir_sign[3] = {
          ray_dirs[ray*3] < 0.0f ? 1 : 0,
          ray_dirs[ray*3+1] < 0.0f ? 1 : 0,
          ray_dirs[ray*3+2] < 0.0f ? 1 : 0
        };
        
        if (IntersectRayAABB(&tmin, &tmax, 0.0f, 1e30f,
                            &aabb_mins[aabb*3], &aabb_maxs[aabb*3],
                            ray_org, ray_inv_dir, dir_sign)) {
          hit_count++;
        }
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Original implementation:\n";
    std::cout << "  Time: " << duration.count() << " microseconds\n";
    std::cout << "  Hits: " << hit_count << "\n";
    std::cout << "  Rate: " << (static_cast<double>(NUM_RAYS * NUM_AABBS) / duration.count()) << " Mrays/sec\n\n";
  }

#ifdef NANORT_ENABLE_SSE2
  // Benchmark SIMD implementation
  {
    auto start = std::chrono::high_resolution_clock::now();
    int hit_count = 0;
    
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      for (int aabb = 0; aabb < NUM_AABBS; aabb++) {
        float tmin, tmax;
        
        if (IntersectRayAABB_SIMD_SSE2(&tmin, &tmax, 0.0f, 1e30f,
                                      &aabb_mins[aabb*3], &aabb_maxs[aabb*3],
                                      &ray_orgs[ray*3], &ray_inv_dirs[ray*3])) {
          hit_count++;
        }
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "SIMD SSE2 implementation:\n";
    std::cout << "  Time: " << duration.count() << " microseconds\n";
    std::cout << "  Hits: " << hit_count << "\n";
    std::cout << "  Rate: " << (static_cast<double>(NUM_RAYS * NUM_AABBS) / duration.count()) << " Mrays/sec\n\n";
  }
#endif

#ifdef NANORT_ENABLE_AVX2  
  // Benchmark AVX2 implementation
  {
    auto start = std::chrono::high_resolution_clock::now();
    int hit_count = 0;
    
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      for (int aabb = 0; aabb < NUM_AABBS; aabb++) {
        float tmin, tmax;
        
        if (IntersectRayAABB_SIMD_AVX2(&tmin, &tmax, 0.0f, 1e30f,
                                       &aabb_mins[aabb*3], &aabb_maxs[aabb*3],
                                       &ray_orgs[ray*3], &ray_inv_dirs[ray*3])) {
          hit_count++;
        }
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "SIMD AVX2 implementation:\n";
    std::cout << "  Time: " << duration.count() << " microseconds\n";
    std::cout << "  Hits: " << hit_count << "\n";
    std::cout << "  Rate: " << (static_cast<double>(NUM_RAYS * NUM_AABBS) / duration.count()) << " Mrays/sec\n\n";
  }
#endif

#ifdef NANORT_ENABLE_NEON
  // Benchmark NEON implementation
  {
    auto start = std::chrono::high_resolution_clock::now();
    int hit_count = 0;
    
    for (int ray = 0; ray < NUM_RAYS; ray++) {
      for (int aabb = 0; aabb < NUM_AABBS; aabb++) {
        float tmin, tmax;
        
        if (IntersectRayAABB_SIMD_NEON(&tmin, &tmax, 0.0f, 1e30f,
                                       &aabb_mins[aabb*3], &aabb_maxs[aabb*3],
                                       &ray_orgs[ray*3], &ray_inv_dirs[ray*3])) {
          hit_count++;
        }
      }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "SIMD NEON implementation:\n";
    std::cout << "  Time: " << duration.count() << " microseconds\n";
    std::cout << "  Hits: " << hit_count << "\n";
    std::cout << "  Rate: " << (static_cast<double>(NUM_RAYS * NUM_AABBS) / duration.count()) << " Mrays/sec\n\n";
  }
#endif
}

int main() {
  std::cout << "NanoRT SIMD Optimization Benchmark\n";
  std::cout << "==================================\n\n";
  
  benchmark_ray_aabb_intersection();
  
  return 0;
}