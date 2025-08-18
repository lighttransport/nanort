const { NanoRTWebGPU, createSimpleScene, createRay } = require('./nanort-webgpu');

class NanoRTRenderer {
    constructor() {
        this.webgpu = new NanoRTWebGPU();
        this.scene = null;
    }

    async initialize() {
        await this.webgpu.initialize();
    }

    setScene(bvhNodes, triangles, primitiveIndices) {
        this.scene = { bvhNodes, triangles, primitiveIndices };
    }

    loadSimpleScene() {
        this.scene = createSimpleScene();
    }

    async renderRays(rays) {
        if (!this.scene) {
            throw new Error('No scene loaded. Call setScene() or loadSimpleScene() first.');
        }

        return await this.webgpu.traverseRays(
            this.scene.bvhNodes,
            this.scene.triangles,
            this.scene.primitiveIndices,
            rays
        );
    }

    async renderImage(width, height, camera) {
        const rays = this.generateCameraRays(width, height, camera);
        const results = await this.renderRays(rays);
        
        const image = new Array(height);
        for (let y = 0; y < height; y++) {
            image[y] = new Array(width);
            for (let x = 0; x < width; x++) {
                const index = y * width + x;
                const result = results[index];
                
                if (result.hit) {
                    const shade = Math.max(0.1, 1.0 - result.t * 0.1);
                    image[y][x] = [shade, shade, shade];
                } else {
                    image[y][x] = [0.0, 0.0, 0.0];
                }
            }
        }
        
        return image;
    }

    generateCameraRays(width, height, camera = {}) {
        const {
            position = [0, 0, -5],
            target = [0, 0, 0],
            up = [0, 1, 0],
            fov = 45
        } = camera;

        const rays = [];
        const aspect = width / height;
        const fovRad = (fov * Math.PI) / 180;
        const tanHalfFov = Math.tan(fovRad / 2);

        const w = this.normalize(this.subtract(position, target));
        const u = this.normalize(this.cross(up, w));
        const v = this.cross(w, u);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const px = (2 * (x + 0.5) / width - 1) * aspect * tanHalfFov;
                const py = (1 - 2 * (y + 0.5) / height) * tanHalfFov;

                const direction = this.normalize([
                    px * u[0] + py * v[0] - w[0],
                    px * u[1] + py * v[1] - w[1],
                    px * u[2] + py * v[2] - w[2]
                ]);

                rays.push(createRay(position, direction));
            }
        }

        return rays;
    }

    normalize(v) {
        const length = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return [v[0] / length, v[1] / length, v[2] / length];
    }

    subtract(a, b) {
        return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    }

    cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    destroy() {
        this.webgpu.destroy();
    }
}

function saveImagePPM(image, filename) {
    const fs = require('fs');
    const height = image.length;
    const width = image[0].length;
    
    let ppm = `P3\n${width} ${height}\n255\n`;
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = image[y][x];
            const r = Math.floor(Math.min(255, Math.max(0, pixel[0] * 255)));
            const g = Math.floor(Math.min(255, Math.max(0, pixel[1] * 255)));
            const b = Math.floor(Math.min(255, Math.max(0, pixel[2] * 255)));
            ppm += `${r} ${g} ${b} `;
        }
        ppm += '\n';
    }
    
    fs.writeFileSync(filename, ppm);
}

module.exports = {
    NanoRTRenderer,
    NanoRTWebGPU,
    createSimpleScene,
    createRay,
    saveImagePPM
};