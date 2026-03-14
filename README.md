# IgnisRenderer

![Version](https://img.shields.io/badge/version-1.0.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite)

IgnisRenderer is a high-performance 3D rendering engine built from the ground up in TypeScript. It features a unique multi-backend architecture, allowing for flexible rendering across CPU (Software), WebGL, and WebGPU hardware pipelines.

[Live Demo](https://ignis-renderer-demo.netlify.app/)

![IgnisRenderer Screenshot](./assets/screenshot.png)

## 🚀 Key Features

- **Triple-Backend Support**:
    - **SoftwareBackend**: A high-efficiency CPU rasterizer implementing custom PBR shading without hardware acceleration dependencies.
    - **WebGPUBackend**: For next-gen hardware performance, featuring an advanced post-processing graph and WGSL shaders.
    - **WebGLBackend**: Modern V1 implementation ensuring broad device compatibility.
- **Advanced Rendering Pipeline**:
    - Physically Based Rendering (PBR) with full IBL (Image-Based Lighting) support.
    - Real-time Shadowing system featuring PCSS and Cascaded Shadow Maps (CSM).
    - Post-Processing Suite: SSR, SSAO, TAA, FXAA, and Volumetric Lighting effects.
- **Deep Simulation Integration**:
    - **Animation**: Skeletal systems with complex blend trees and hierarchical state machines.
    - **Physics**: Adapter-based integration for Rapier3D and Ammo.js with real-time scene synchronization.
    - **Particles**: Temporal simulation stage for high-density visual effects.
- **High-Performance ECS Foundation**:
    - Core logic backed by a custom Entity Component System (ECS) for efficient data locality.
    - `Node`-based hierarchy as a compatibility facade over the underlying ECS architecture.
- **Professional Math Library**: Optimized linear algebra implementation targeting zero-allocation hot paths.

## 📁 Project Structure

The codebase is organized into modular directories, separating core abstractions from implementation-specific backends and simulation logic.

### `src/` - Source Code
- **`core/`**: Fundamental abstractions including Scene Graph management (`Node`, `Scene`) and Resource types (`Texture`).
- **`ecs/`**: The underlying Entity Component System managing high-performance data storage and entity state.
- **`renderers/`**: Multi-backend implementations. Contains specialized logic for Software, WebGL, and WebGPU pipelines.
- **`pipeline/`**: The execution frame graph, responsible for feature resolution, frame planning, and draw call preparation.
- **`shaders/`**: Centralized shader source repository containing WGSL and GLSL modules for all rendering backends.
- **`animation/`**: Skeletal animation logic, state machines, and blend tree implementations.
- **`physics/`**: The physics simulation layer and adapters for external engines (Rapier3D, Ammo.js).
- **`particles/`**: Specialized simulation stage for maintaining and updating particle systems.
- **`maths/`**: A optimized math library focused on performance and memory efficiency.
- **`loaders/`**: Resource acquisition logic, featuring a robust glTF 2.0 parser.
- **`materials/` & `meshes/`**: Assets management for PBR shading and geometric data representation.
- **`cameras/` & `lights/`**: Specialized scene objects for view control and illumination.

### `tests/` - Testing Suite
- Comprehensive suite of over 50 automated tests covering lighting accuracy, physics consistency, animation state blending, and backend-specific feature verification.

## 🏗️ Architecture Overview

IgnisRenderer utilizes a data-driven design pattern to ensure scalability across diverse rendering targets and simulation complexities.

### 1. Data-Oriented Design (ECS)
The engine has transitioned to an **ECS-first architecture**. While traditional scene graph traversal is supported via the `Node` facade, the internal state is managed by the `ECSWorld`. This allows for high-efficiency queries and batch updates, particularly beneficial for complex simulations like particles and physics.

### 2. Multi-Stage Execution Pipeline
Each frame undergoes a rigid sequence of operations:
- **Feature Resolution**: The `FeatureResolver` inspects the scene to determine active requirements (e.g., specific shadow types, IBL requirements, or post-processing passes).
- **Simulation Flow**: Animation, physics, and particle stages update their respective states in a unified timeline (measured strictly in seconds).
- **Prepared Scene Construction**: The `PreparedSceneBuilder` collects scene data into optimized draw packets, decoupling the scene graph from backend dispatches.
- **Backend-Agnostic Dispatch**: Commands are encoded through a unified interface, allowing the engine to switch between CPU and GPU rendering seamlessly.

### 3. Backend Strategy
- **Software Rasterizer**: Employs advanced CPU-side algorithms for sub-pixel accuracy and PBR shading, serving as both a fallback and a benchmark for hardware backends.
- **Modern Hardware Backends**: The WebGPU implementation leverages a state-of-the-art post-processing graph, while the WebGL V1 backend provides a stable bridge for legacy devices.

### 4. Mathematical Optimization
To maintain performance in the Software backend and reduce GC pressure during hardware command encoding, the `maths` library uses pre-allocated objects and in-place mutations. The coordinate system is right-handed (Y-up) with a standard NDC range.

## 🧪 Development & Testing

The project maintains architectural integrity through a strict testing regime. Developers can run specialized suites for lighting, geometry, animation, and backend capabilities:
- **Backend Verification**: Ensures feature parity between Software, WebGL, and WebGPU implementations.
- **Simulation Accuracy**: Validates physics adapter contracts and skeletal blend tree results.
- **Resource Integrity**: Checks glTF loading consistency and PBR texture mapping.

All simulation logic is standardized to use seconds for delta time, ensuring deterministic behavior across different hardware configurations.

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.
