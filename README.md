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

## 🏗️ Architecture Overview

The project is structured with a modular architecture to support the backend-agnostic rendering pipeline and comprehensive simulation stages. 

### 1. Scene Graph & Core (`src/core` & `src/ecs`)
The foundation of IgnisRenderer revolves around the `Node` system, which manages bounding volumes and hierarchical transforms. Shared `MeshAsset`s and individual `MeshInstance`s separate geometry data from its spatial representation. The `Scene` object serves as the root container and manages rendering layers and resource loading via components like `GLTFLoader`.

### 2. Simulation Stages (`src/simulation`)
Integrated directly into the frame execution flow, the simulation is broken into focused layers:
- **Animation (`src/animation`)**: Processes clips, skeletal structures, blend trees, and state machines. Works entirely in seconds to maintain engine-wide consistency.
- **Physics (`src/physics`)**: An adapter-based system (binding Rapier3D or Ammo.js) updating transformations of `PhysicsBodyNode`s. Ensures the rendering engine remains engine-agnostic.
- **Particles (`src/particles`)**: Computes temporal properties for visual effects before they are passed to the renderer.

### 3. Execution Pipeline (`src/pipeline`)
The engine dynamically plans each frame in discrete stages:
- **Feature Resolution**: Determines required shading models (IBL, post-processing options, multiple shadow types).
- **Simulation**: Propagates time updates to particles, physics adapters, and animation mixers.
- **Prepared Scene Construction**: Traverses the `Scene` graph to collect draw packets indexed by `MeshInstance` and `MeshAsset`, optimizing batch dispatches.
- **Backend Dispatch**: Passes standardized draw calls to the active renderer.

### 4. Mathematical Foundation (`src/maths`)
Designed strictly for zero-allocation loops, which is critically important for `SoftwareBackend` rasterization paths. Math primitives use right-handed coordinates (Y-up, Z-towards viewer) and perform in-place mutations to prevent garbage collection spikes in hot loops.

### 5. Render Backends (`src/renderers` & `src/shaders`)
- **Software Backend**: Driven by CPU algorithms, converting vector graphics and shader logic into pure per-pixel calculations without WebGL/WebGPU overhead.
- **WebGPU Backend**: Hardware-optimized dispatch mechanism bridging TypeScript constructs with corresponding WGSL shader modules. Features a complex post-processing graph supporting complex multi-pass rendering like SSR.
- **WebGL Backend**: Included as a legacy/stub backend for fallback scenarios.

## 🧪 Development & Testing

The development environment favors extensionless relative imports for source files and isolates test runners using `tsx`. IgnisRenderer maintains high coverage across its complex systems with specialized test setups tailored to distinct architectural units:
- **Lighting**: Verifies PBR and Spherical Harmonics behavior.
- **Animation & Physics**: Checks skeletal states, blend trees, and adapter constraints.
- **WebGPU**: Validates the bridge and post-processing graph compilation.
- **Geometry**: Tests model formatting and winding orders.
All simulation logic scales identically by standardizing the delta time exclusively in seconds.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
