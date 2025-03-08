#include "nanort.h"
#include <iostream>
#include <vector>
#include <random>

int main(int argc, char ** argv) {

    // Number of triangles and rays
    int num_triangles = 10000;
    int num_rays = 10000;

    if (argc > 1) {
      num_triangles = std::atoi(argv[1]);
    }

    if (argc > 2) {
      num_rays = std::atoi(argv[2]);
    }

    std::cout << "# of triangles: " << num_triangles << "\n";
    std::cout << "# of rays: " << num_rays << "\n";

    // Generate random triangles
    std::vector<float> vertices(num_triangles * 9); // 3 vertices per triangle, 3 coordinates per vertex
    std::vector<unsigned int> faces(num_triangles * 3); // 3 indices per triangle

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < num_triangles * 9; ++i) {
        vertices[i] = dis(gen);
    }

    for (int i = 0; i < num_triangles * 3; ++i) {
        faces[i] = i;
    }

    // Generate random rays
    std::vector<nanort::Ray<float>> rays(num_rays);
    for (int i = 0; i < num_rays; ++i) {
        rays[i].org[0] = dis(gen);
        rays[i].org[1] = dis(gen);
        rays[i].org[2] = dis(gen);
        rays[i].dir[0] = dis(gen);
        rays[i].dir[1] = dis(gen);
        rays[i].dir[2] = dis(gen);
        rays[i].min_t = 0.0f;
        rays[i].max_t = 1.0e+30f;
    }

    // Build BVH
    nanort::BVHAccel<float> accel;
    nanort::TriangleMesh<float> triangle_mesh(vertices.data(), faces.data(), sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(vertices.data(), faces.data(), sizeof(float) * 3);
    nanort::BVHBuildOptions<float> build_options;
    accel.Build(num_triangles, triangle_mesh, triangle_pred, build_options);

    // Traverse rays and report AABB and leaf tests
    nanort::BVHTraceOptions trace_options;
    size_t total_aabb_tests = 0;
    size_t total_leaf_tests = 0;

    for (const auto& ray : rays) {
        nanort::TriangleIntersector<float> intersector(vertices.data(), faces.data(), sizeof(float) * 3);
        nanort::TriangleIntersection<float> isect;
        size_t aabb_tests = 0;
        size_t leaf_tests = 0;
        accel.Traverse(ray, intersector, &isect, trace_options, &aabb_tests, &leaf_tests);
        total_aabb_tests += aabb_tests;
        total_leaf_tests += leaf_tests;
    }

    std::cout << "Benchmark completed." << std::endl;
    std::cout << "Average AABB tests per ray: " << static_cast<double>(total_aabb_tests) / num_rays << std::endl;
    std::cout << "Average leaf tests per ray: " << static_cast<double>(total_leaf_tests) / num_rays << std::endl;

    return 0;
}
