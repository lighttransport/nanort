// WebGPU compute shader for BVH traversal
// Based on NanoRT traversal algorithm

struct Ray {
    origin: vec3f,
    min_t: f32,
    direction: vec3f,
    max_t: f32
}

struct BVHNode {
    bmin: vec3f,
    flag: u32,  // 1 = leaf, 0 = branch
    bmax: vec3f,
    axis: u32,
    data0: u32,  // child[0] or npoints for leaf
    data1: u32   // child[1] or index for leaf
}

struct IntersectionResult {
    hit: u32,      // 0 = no hit, 1 = hit
    t: f32,        // hit distance
    primitive_id: u32,
    u: f32,        // barycentric u
    v: f32         // barycentric v
}

struct Triangle {
    v0: vec3f,
    _pad0: f32,
    v1: vec3f,
    _pad1: f32,
    v2: vec3f,
    _pad2: f32
}

@group(0) @binding(0) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> primitive_indices: array<u32>;
@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<storage, read_write> results: array<IntersectionResult>;

const MAX_STACK_DEPTH: u32 = 64u;

fn safe_inverse(v: vec3f) -> vec3f {
    let eps = 1e-8;
    return vec3f(
        select(1.0 / v.x, 1e30, abs(v.x) < eps),
        select(1.0 / v.y, 1e30, abs(v.y) < eps),
        select(1.0 / v.z, 1e30, abs(v.z) < eps)
    );
}

fn intersect_ray_aabb(ray_org: vec3f, ray_inv_dir: vec3f, dir_sign: vec3<u32>, 
                      bmin: vec3f, bmax: vec3f, ray_min_t: f32, ray_max_t: f32) -> vec2f {
    let bounds = array<vec3f, 2>(bmin, bmax);
    
    let t_min = (bounds[dir_sign.x].x - ray_org.x) * ray_inv_dir.x;
    let t_max = (bounds[1u - dir_sign.x].x - ray_org.x) * ray_inv_dir.x;
    
    let ty_min = (bounds[dir_sign.y].y - ray_org.y) * ray_inv_dir.y;
    let ty_max = (bounds[1u - dir_sign.y].y - ray_org.y) * ray_inv_dir.y;
    
    var t_enter = max(t_min, ty_min);
    var t_exit = min(t_max, ty_max);
    
    let tz_min = (bounds[dir_sign.z].z - ray_org.z) * ray_inv_dir.z;
    let tz_max = (bounds[1u - dir_sign.z].z - ray_org.z) * ray_inv_dir.z;
    
    t_enter = max(t_enter, tz_min);
    t_exit = min(t_exit, tz_max);
    
    t_enter = max(t_enter, ray_min_t);
    t_exit = min(t_exit, ray_max_t);
    
    return vec2f(t_enter, t_exit);
}

fn intersect_ray_triangle(ray_org: vec3f, ray_dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> vec4f {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);
    
    if (abs(a) < 1e-8) {
        return vec4f(-1.0, 0.0, 0.0, 0.0); // No intersection
    }
    
    let f = 1.0 / a;
    let s = ray_org - v0;
    let u = f * dot(s, h);
    
    if (u < 0.0 || u > 1.0) {
        return vec4f(-1.0, 0.0, 0.0, 0.0);
    }
    
    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    
    if (v < 0.0 || u + v > 1.0) {
        return vec4f(-1.0, 0.0, 0.0, 0.0);
    }
    
    let t = f * dot(edge2, q);
    
    if (t > 1e-8) {
        return vec4f(t, u, v, 1.0); // Hit
    }
    
    return vec4f(-1.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if (ray_index >= arrayLength(&rays)) {
        return;
    }
    
    let ray = rays[ray_index];
    var result = IntersectionResult();
    result.hit = 0u;
    result.t = ray.max_t;
    result.primitive_id = 0xFFFFFFFFu;
    
    if (arrayLength(&bvh_nodes) == 0u) {
        results[ray_index] = result;
        return;
    }
    
    let ray_inv_dir = safe_inverse(ray.direction);
    let dir_sign = vec3<u32>(
        select(0u, 1u, ray.direction.x < 0.0),
        select(0u, 1u, ray.direction.y < 0.0),
        select(0u, 1u, ray.direction.z < 0.0)
    );
    
    var node_stack: array<u32, MAX_STACK_DEPTH>;
    var stack_index = 0u;
    node_stack[0] = 0u;
    
    var hit_t = ray.max_t;
    
    while (stack_index < MAX_STACK_DEPTH) {
        let current_index = node_stack[stack_index];
        if (stack_index == 0u && current_index == 0xFFFFFFFFu) {
            break;
        }
        stack_index = stack_index - 1u;
        
        if (current_index >= arrayLength(&bvh_nodes)) {
            continue;
        }
        
        let node = bvh_nodes[current_index];
        let aabb_result = intersect_ray_aabb(ray.origin, ray_inv_dir, dir_sign, 
                                           node.bmin, node.bmax, ray.min_t, hit_t);
        
        if (aabb_result.x <= aabb_result.y && aabb_result.y >= ray.min_t) {
            if (node.flag == 0u) {
                // Branch node
                let order_near = dir_sign[node.axis];
                let order_far = 1u - order_near;
                
                if (stack_index + 2u < MAX_STACK_DEPTH) {
                    stack_index = stack_index + 1u;
                    node_stack[stack_index] = select(node.data0, node.data1, order_far == 1u);
                    stack_index = stack_index + 1u;
                    node_stack[stack_index] = select(node.data0, node.data1, order_near == 1u);
                }
            } else {
                // Leaf node
                let num_primitives = node.data0;
                let primitive_offset = node.data1;
                
                for (var i = 0u; i < num_primitives; i = i + 1u) {
                    let prim_index = primitive_indices[primitive_offset + i];
                    if (prim_index >= arrayLength(&triangles)) {
                        continue;
                    }
                    
                    let triangle = triangles[prim_index];
                    let tri_result = intersect_ray_triangle(ray.origin, ray.direction,
                                                          triangle.v0, triangle.v1, triangle.v2);
                    
                    if (tri_result.w > 0.0 && tri_result.x >= ray.min_t && tri_result.x < hit_t) {
                        hit_t = tri_result.x;
                        result.hit = 1u;
                        result.t = tri_result.x;
                        result.primitive_id = prim_index;
                        result.u = tri_result.y;
                        result.v = tri_result.z;
                    }
                }
            }
        }
        
        if (stack_index == 0u) {
            node_stack[0] = 0xFFFFFFFFu;
        }
    }
    
    results[ray_index] = result;
}