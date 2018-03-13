// g++ -O0 -std=c++11 -Wall main_bug_1.cpp  -o main_bug_1
#include "nanort.h"
#include <iostream>

typedef double real;

int main(int argc, char * argv[])
{
    std::cout << "This program exposes a possible accuracy bug with nanort." << std::endl<<std::flush;
    std::cout << "A slight variation on direction[0] makes nanort misses a real intersection." << std::endl<<std::flush;
    std::cout << "Called without argument, we run a normal case where a ray creates an intersection on a single triangle" << std::endl<<std::flush;
    std::cout << "Called with one argument, we slightty changes the direction and have a no intersection result, that is not normal" << std::endl<<std::flush;
    bool activate_precision_bug = false;
    if (argc>1)
    {
        activate_precision_bug = true;
    }
    real vertices[9];
    unsigned int triangles[3] ={0,1,2};

    const real xMin=-1.0, xMax=+1.0;
    const real zMin=-3.0, zMax=+3.0;
    vertices[3 * 0] = xMax; vertices[3 * 0 + 1] = 2.0; vertices[3 * 0 + 2] = zMin;
    vertices[3 * 1] = xMin; vertices[3 * 1 + 1] = 2.0; vertices[3 * 1 + 2] = zMin;
    vertices[3 * 2] = xMax; vertices[3 * 2 + 1] = 2.0; vertices[3 * 2 + 2] = zMax;

    real origins[3];
    real directions[3];

    origins[0] = -0.36; origins[1] = +7.93890843; origins[2] = 1.2160368;
    directions[1] = -8.66025404e-01; directions[2] = -0.5;
    directions[0] = 0.0;
    if (activate_precision_bug)
    {
        directions[0] = -5.30287619e-17;
    }
    std::cout << "directions[0] = " << directions[0] << std::endl;

    nanort::BVHBuildOptions<real> build_options; // Use default option
    nanort::TriangleMesh<real> triangle_mesh(vertices, triangles, sizeof(real) * 3);
    nanort::TriangleSAHPred<real> triangle_pred(vertices, triangles, sizeof(real) * 3);
    nanort::BVHAccel<real> accel;
    build_options.cache_bbox = true;
    int ret = accel.Build((size_t) 1, triangle_mesh, triangle_pred, build_options);
    assert(ret);
    nanort::Ray<real> ray;
    nanort::TriangleIntersector<real, nanort::TriangleIntersection<real> > triangle_intersector(vertices, triangles, sizeof(real) * 3);
    nanort::TriangleIntersection<real> isect;

    ray.org[0] = origins[0];
    ray.org[1] = origins[1];
    ray.org[2] = origins[2];


    const real length = sqrt(directions[0] * directions[0] +
                             directions[1] * directions[1] +
                             directions[2] * directions[2]);
    ray.dir[0] = directions[0]/length;
    ray.dir[1] = directions[1]/length;
    ray.dir[2] = directions[2]/length;

    ray.min_t = 0.0;
    ray.max_t =  1.0e+30;

    const bool hit = accel.Traverse(ray, triangle_intersector, &isect);
    if (hit)
    {
        std::cout << "We have the expected result" << std::endl<<std::flush;
        std::cout << "Intersection isect.u =" << isect.u << " v = " << isect.v << std::endl<<std::flush;
    }
    else
    {
        std::cout << "No intersection detected" << std::endl<<std::flush;
        std::cout << "We get the wrong result" << std::endl<<std::flush;

    }
    return 0;
}
