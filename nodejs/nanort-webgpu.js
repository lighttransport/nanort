const fs = require('fs');
const path = require('path');

class NanoRTWebGPU {
    constructor() {
        this.device = null;
        this.commandEncoder = null;
        this.computePipeline = null;
        this.bindGroup = null;
        this.isInitialized = false;
    }

    async initialize() {
        if (this.isInitialized) return;

        try {
            const { GPU } = require('webgpu');
            const gpu = new GPU();
            
            const adapter = await gpu.requestAdapter();
            if (!adapter) {
                throw new Error('Failed to get WebGPU adapter');
            }

            this.device = await adapter.requestDevice();
            
            const shaderSource = fs.readFileSync(
                path.join(__dirname, 'bvh-traversal.wgsl'), 
                'utf8'
            );
            
            const shaderModule = this.device.createShaderModule({
                code: shaderSource,
            });

            this.computePipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: shaderModule,
                    entryPoint: 'main',
                },
            });

            this.isInitialized = true;
        } catch (error) {
            throw new Error(`Failed to initialize WebGPU: ${error.message}`);
        }
    }

    createBuffer(data, usage) {
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: usage,
            mappedAtCreation: true,
        });

        if (data instanceof Float32Array) {
            new Float32Array(buffer.getMappedRange()).set(data);
        } else if (data instanceof Uint32Array) {
            new Uint32Array(buffer.getMappedRange()).set(data);
        } else {
            new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data));
        }

        buffer.unmap();
        return buffer;
    }

    createBVHBuffer(nodes) {
        const nodeSize = 8 * 4; // 8 floats/uints per node
        const bufferData = new Float32Array(nodes.length * 8);
        
        for (let i = 0; i < nodes.length; i++) {
            const offset = i * 8;
            const node = nodes[i];
            
            bufferData[offset + 0] = node.bmin[0];
            bufferData[offset + 1] = node.bmin[1];
            bufferData[offset + 2] = node.bmin[2];
            bufferData[offset + 3] = node.flag;
            
            bufferData[offset + 4] = node.bmax[0];
            bufferData[offset + 5] = node.bmax[1];
            bufferData[offset + 6] = node.bmax[2];
            bufferData[offset + 7] = node.axis;
            
            const view = new Uint32Array(bufferData.buffer, (offset + 8) * 4, 2);
            view[0] = node.data[0];
            view[1] = node.data[1];
        }

        return this.createBuffer(bufferData, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }

    createTriangleBuffer(triangles) {
        const triangleSize = 12; // 3 vec3f + padding = 12 floats per triangle
        const bufferData = new Float32Array(triangles.length * triangleSize);
        
        for (let i = 0; i < triangles.length; i++) {
            const offset = i * triangleSize;
            const tri = triangles[i];
            
            bufferData[offset + 0] = tri.v0[0];
            bufferData[offset + 1] = tri.v0[1];
            bufferData[offset + 2] = tri.v0[2];
            bufferData[offset + 3] = 0.0; // padding
            
            bufferData[offset + 4] = tri.v1[0];
            bufferData[offset + 5] = tri.v1[1];
            bufferData[offset + 6] = tri.v1[2];
            bufferData[offset + 7] = 0.0; // padding
            
            bufferData[offset + 8] = tri.v2[0];
            bufferData[offset + 9] = tri.v2[1];
            bufferData[offset + 10] = tri.v2[2];
            bufferData[offset + 11] = 0.0; // padding
        }

        return this.createBuffer(bufferData, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }

    createRayBuffer(rays) {
        const raySize = 8; // 2 vec3f + 2 floats = 8 floats per ray
        const bufferData = new Float32Array(rays.length * raySize);
        
        for (let i = 0; i < rays.length; i++) {
            const offset = i * raySize;
            const ray = rays[i];
            
            bufferData[offset + 0] = ray.origin[0];
            bufferData[offset + 1] = ray.origin[1];
            bufferData[offset + 2] = ray.origin[2];
            bufferData[offset + 3] = ray.min_t || 0.0;
            
            bufferData[offset + 4] = ray.direction[0];
            bufferData[offset + 5] = ray.direction[1];
            bufferData[offset + 6] = ray.direction[2];
            bufferData[offset + 7] = ray.max_t || 1e30;
        }

        return this.createBuffer(bufferData, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    }

    createResultBuffer(numRays) {
        const resultSize = 5; // hit(u32), t(f32), primitive_id(u32), u(f32), v(f32)
        const bufferData = new Float32Array(numRays * resultSize);
        bufferData.fill(0);

        return this.createBuffer(bufferData, 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    }

    async traverseRays(bvhNodes, triangles, primitiveIndices, rays) {
        if (!this.isInitialized) {
            await this.initialize();
        }

        const bvhBuffer = this.createBVHBuffer(bvhNodes);
        const triangleBuffer = this.createTriangleBuffer(triangles);
        const indexBuffer = this.createBuffer(new Uint32Array(primitiveIndices), 
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const rayBuffer = this.createRayBuffer(rays);
        const resultBuffer = this.createResultBuffer(rays.length);

        this.bindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: bvhBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: triangleBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: indexBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: rayBuffer,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: resultBuffer,
                    },
                },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        
        passEncoder.setPipeline(this.computePipeline);
        passEncoder.setBindGroup(0, this.bindGroup);
        
        const workgroupCount = Math.ceil(rays.length / 64);
        passEncoder.dispatchWorkgroups(workgroupCount);
        passEncoder.end();

        const copyBuffer = this.device.createBuffer({
            size: resultBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        commandEncoder.copyBufferToBuffer(
            resultBuffer, 0,
            copyBuffer, 0,
            resultBuffer.size
        );

        this.device.queue.submit([commandEncoder.finish()]);

        await copyBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(copyBuffer.getMappedRange());
        
        const results = [];
        const resultSize = 5;
        
        for (let i = 0; i < rays.length; i++) {
            const offset = i * resultSize;
            const hitView = new Uint32Array(resultData.buffer, offset * 4, 1);
            const primitiveView = new Uint32Array(resultData.buffer, (offset + 2) * 4, 1);
            
            results.push({
                hit: hitView[0] !== 0,
                t: resultData[offset + 1],
                primitive_id: primitiveView[0],
                u: resultData[offset + 3],
                v: resultData[offset + 4]
            });
        }

        copyBuffer.unmap();
        
        bvhBuffer.destroy();
        triangleBuffer.destroy();
        indexBuffer.destroy();
        rayBuffer.destroy();
        resultBuffer.destroy();
        copyBuffer.destroy();

        return results;
    }

    destroy() {
        if (this.device) {
            this.device.destroy();
            this.device = null;
        }
        this.isInitialized = false;
    }
}

function createSimpleScene() {
    const triangles = [
        {
            v0: [0.0, 1.0, 0.0],
            v1: [-1.0, -1.0, 0.0],
            v2: [1.0, -1.0, 0.0]
        }
    ];

    const bvhNodes = [
        {
            bmin: [-1.0, -1.0, 0.0],
            bmax: [1.0, 1.0, 0.0],
            flag: 1, // leaf
            axis: 0,
            data: [1, 0] // 1 primitive at index 0
        }
    ];

    const primitiveIndices = [0];

    return { bvhNodes, triangles, primitiveIndices };
}

function createRay(origin, direction, min_t = 0.0, max_t = 1e30) {
    const length = Math.sqrt(
        direction[0] * direction[0] + 
        direction[1] * direction[1] + 
        direction[2] * direction[2]
    );
    
    return {
        origin: origin,
        direction: [
            direction[0] / length,
            direction[1] / length,
            direction[2] / length
        ],
        min_t: min_t,
        max_t: max_t
    };
}

module.exports = {
    NanoRTWebGPU,
    createSimpleScene,
    createRay
};