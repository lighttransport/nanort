const { NanoRTRenderer, createRay, createSimpleScene } = require('./index');

function assert(condition, message) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

function assertApproxEqual(a, b, tolerance = 1e-6, message = '') {
    const diff = Math.abs(a - b);
    assert(diff < tolerance, `${message}: expected ${a} ≈ ${b}, diff: ${diff}`);
}

async function testBasicRayTriangleIntersection() {
    console.log('Testing basic ray-triangle intersection...');
    
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    renderer.loadSimpleScene();
    
    const ray = createRay([0, 0, -1], [0, 0, 1]);
    const results = await renderer.renderRays([ray]);
    
    assert(results.length === 1, 'Should return one result');
    assert(results[0].hit === true, 'Ray should hit triangle');
    assertApproxEqual(results[0].t, 1.0, 1e-3, 'Hit distance should be approximately 1.0');
    
    renderer.destroy();
    console.log('✓ Basic ray-triangle intersection test passed');
}

async function testMissedRay() {
    console.log('Testing missed ray...');
    
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    renderer.loadSimpleScene();
    
    const ray = createRay([2, 0, -1], [0, 0, 1]);
    const results = await renderer.renderRays([ray]);
    
    assert(results.length === 1, 'Should return one result');
    assert(results[0].hit === false, 'Ray should miss triangle');
    
    renderer.destroy();
    console.log('✓ Missed ray test passed');
}

async function testMultipleRays() {
    console.log('Testing multiple rays...');
    
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    renderer.loadSimpleScene();
    
    const rays = [
        createRay([0, 0, -1], [0, 0, 1]),     // hit
        createRay([2, 0, -1], [0, 0, 1]),     // miss
        createRay([-0.5, 0, -1], [0, 0, 1]),  // hit
        createRay([0, 2, -1], [0, 0, 1])      // miss
    ];
    
    const results = await renderer.renderRays(rays);
    
    assert(results.length === 4, 'Should return four results');
    assert(results[0].hit === true, 'First ray should hit');
    assert(results[1].hit === false, 'Second ray should miss');
    assert(results[2].hit === true, 'Third ray should hit');
    assert(results[3].hit === false, 'Fourth ray should miss');
    
    renderer.destroy();
    console.log('✓ Multiple rays test passed');
}

async function testCustomScene() {
    console.log('Testing custom scene...');
    
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    
    const triangles = [
        {
            v0: [-1, -1, 0],
            v1: [1, -1, 0],
            v2: [0, 1, 0]
        },
        {
            v0: [0, -1, 1],
            v1: [2, -1, 1],
            v2: [1, 1, 1]
        }
    ];
    
    const bvhNodes = [
        {
            bmin: [-1, -1, 0],
            bmax: [2, 1, 1],
            flag: 0, // branch
            axis: 2, // z-axis
            data: [1, 2]
        },
        {
            bmin: [-1, -1, 0],
            bmax: [1, 1, 0],
            flag: 1, // leaf
            axis: 0,
            data: [1, 0] // 1 primitive at index 0
        },
        {
            bmin: [0, -1, 1],
            bmax: [2, 1, 1],
            flag: 1, // leaf
            axis: 0,
            data: [1, 1] // 1 primitive at index 1
        }
    ];
    
    const primitiveIndices = [0, 1];
    
    renderer.setScene(bvhNodes, triangles, primitiveIndices);
    
    const rays = [
        createRay([0, 0, -1], [0, 0, 1]),  // hit first triangle
        createRay([1, 0, 0], [0, 0, 1])    // hit second triangle
    ];
    
    const results = await renderer.renderRays(rays);
    
    assert(results.length === 2, 'Should return two results');
    assert(results[0].hit === true, 'First ray should hit first triangle');
    assert(results[1].hit === true, 'Second ray should hit second triangle');
    assertApproxEqual(results[0].t, 1.0, 1e-3, 'First hit distance');
    assertApproxEqual(results[1].t, 1.0, 1e-3, 'Second hit distance');
    
    renderer.destroy();
    console.log('✓ Custom scene test passed');
}

async function testImageRendering() {
    console.log('Testing image rendering...');
    
    const renderer = new NanoRTRenderer();
    await renderer.initialize();
    renderer.loadSimpleScene();
    
    const width = 4;
    const height = 4;
    
    const image = await renderer.renderImage(width, height);
    
    assert(image.length === height, 'Image should have correct height');
    assert(image[0].length === width, 'Image should have correct width');
    
    let hitCount = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = image[y][x];
            assert(Array.isArray(pixel), 'Pixel should be an array');
            assert(pixel.length === 3, 'Pixel should have 3 components');
            
            if (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) {
                hitCount++;
            }
        }
    }
    
    assert(hitCount > 0, 'Some pixels should be lit');
    
    renderer.destroy();
    console.log('✓ Image rendering test passed');
}

async function runAllTests() {
    console.log('NanoRT WebGPU Tests');
    console.log('===================\n');
    
    try {
        await testBasicRayTriangleIntersection();
        await testMissedRay();
        await testMultipleRays();
        await testCustomScene();
        await testImageRendering();
        
        console.log('\n✅ All tests passed!');
        
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

if (require.main === module) {
    runAllTests();
}