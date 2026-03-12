# IgnisRenderer

![Version](https://img.shields.io/badge/version-1.0.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite)

IgnisRenderer is a high-performance 3D rendering engine built from the ground up in TypeScript. It features a unique dual-backend architecture, allowing for identical scene rendering across both CPU (Software) and GPU (WebGPU) pipelines.

[Live Demo](https://ignis-renderer-demo.netlify.app/)

![IgnisRenderer Screenshot](./assets/screenshot.png)

## 🚀 Key Features

- **Dual-Backend Support**:
    - **SoftwareBackend**: A complete C++ -inspired CPU rasterizer with multi-threaded potential and custom PBR shading.
    - **WebGPUBackend**: Hardware-accelerated pipeline utilizing modern WGSL shaders.
- **Advanced Rendering Pipeline**:
    - Physically Based Rendering (PBR) with IBL (Image-Based Lighting).
    - Real-time Shadows (PCSS/CSM).
    - Advanced Post-Processing: SSR (Screen Space Reflections), SSAO (Ambient Occlusion), TAA (Temporal Anti-Aliasing), FXAA, and Volumetric Lighting.
- **Deep Simulation Integration**:
    - **Animation System**: Complete skeletal animation, mixers, blend trees (1D/Direct), and hierarchical state machines.
    - **Physics System**: Adapter-based integration for Rapier3D and Ammo.js, syncing physics bodies directly with the scene graph.
    - **Particle System**: Integrated simulation stage for high-performance visual effects.
- **Universal Scene Graph**:
    - `Node`-based hierarchy for transform management.
    - Shared `MeshAsset` resources with per-node `MeshInstance` overrides.
    - Native support for glTF 2.0 (Hierarchy, Materials, Cameras, Lights).
- **Pro-Grade Math Library**: Custom implementation of Vectors, Matrices, Quaternions, and Spherical Harmonics (SH) optimized for zero-allocation loops.

## 📦 Installation

```bash
npm install ignisrenderer
# Optional physics adapters
npm install @dimforge/rapier3d-compat ammo.js
```

## 🛠️ Quick Start

```typescript
import {
    Scene,
    Renderer,
    WebGPUBackend,
    GLTFLoader,
    OrbitCamera,
    AmbientLight,
    DirectionalLight,
    PBRMaterial,
    MeshFactory
} from 'ignisrenderer'

async function initEngine(canvas: HTMLCanvasElement) {
    // 1. Setup Scene & Camera
    const scene = new Scene()
    const camera = new OrbitCamera({ x: 0, y: 10, z: 20 }, 50)
    scene.add(camera)

    // 2. Add Lighting
    scene.add(new AmbientLight({ color: { r: 255, g: 255, b: 255 }, intensity: 0.2 }))
    const sun = new DirectionalLight({ direction: { x: -1, y: -1, z: -1 }, intensity: 1.5 })
    scene.add(sun)

    // 3. Load 3D Models
    const loader = new GLTFLoader()
    const model = await loader.load('./assets/models/character.glb')
    scene.add(model)

    // 4. Initialize Renderer
    const backend = navigator.gpu ? new WebGPUBackend() : new SoftwareBackend()
    const renderer = new Renderer(backend, canvas, camera)
    
    renderer.setScene(scene)
    await renderer.init()

    // 5. Start Render Loop
    function frame(timeMs: number) {
        const deltaTimeSeconds = timeMs / 1000
        renderer.render(deltaTimeSeconds)
        requestAnimationFrame(frame)
    }
    requestAnimationFrame(frame)
}
```

## 🏗️ Architecture Overview

| Module | Description |
| :--- | :--- |
| **src/core** | Scene graph primitives, `Node`, `Scene`, and resource management. |
| **src/pipeline** | Frame planning, feature resolution, and simulation stages. |
| **src/renderers** | Backend implementations (Software, WebGPU, WebGL). |
| **src/animation** | Mixers, Blend Trees, Clips, and State Machines. |
| **src/physics** | Global physics system and external engine adapters. |
| **src/shaders** | Software pixel logic and WebGPU WGSL modules. |
| **src/maths** | Performance-optimized linear algebra library. |

## 🧪 Development

### Build Commands
```bash
npm run dev    # Start dev server
npm run build  # Full production build
npm test       # Run comprehensive test suite
```

### Specialized Tests
IgnisRenderer maintains high coverage across its complex systems:
- `npm run test:lighting` - PBR/SH/Standard lighting verification.
- `npx tsx tests/test_animation_core.mjs` - Skeletal & BlendTree validation.
- `npx tsx tests/test_physics_system_bindings.mjs` - Physics provider contracts.
- `npx tsx tests/test_webgpu_post_graph.mjs` - SSR/SSAO/TAA pipeline checks.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
