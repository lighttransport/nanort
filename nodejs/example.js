const { NanoRTRenderer, saveImagePPM } = require('./index');

async function main() {
    console.log('NanoRT WebGPU Example');
    console.log('====================');

    try {
        const renderer = new NanoRTRenderer();
        
        console.log('Initializing WebGPU...');
        await renderer.initialize();
        
        console.log('Loading simple scene...');
        renderer.loadSimpleScene();
        
        console.log('Rendering 256x256 image...');
        const width = 256;
        const height = 256;
        
        const camera = {
            position: [0, 0, -3],
            target: [0, 0, 0],
            up: [0, 1, 0],
            fov: 45
        };
        
        const startTime = Date.now();
        const image = await renderer.renderImage(width, height, camera);
        const endTime = Date.now();
        
        console.log(`Rendering completed in ${endTime - startTime}ms`);
        
        const outputFile = 'output.ppm';
        saveImagePPM(image, outputFile);
        console.log(`Image saved to ${outputFile}`);
        
        renderer.destroy();
        
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}