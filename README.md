# IgnisRenderer

![Version](https://img.shields.io/badge/version-1.0.2-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite)

IgnisRenderer is a high-performance 3D rendering engine built from the ground up in TypeScript. It features a unique multi-backend architecture, allowing for flexible rendering across CPU (Software), WebGL, and WebGPU hardware pipelines.

[Live Demo](https://ignis-renderer-demo.netlify.app/)

![IgnisRenderer Screenshot](./assets/screenshot.png)

## 🚀 Key Features

- **Triple-Backend Support**:
    - **SoftwareBackend**: A high-efficiency CPU rasterizer implementing custom PBR shading with modular executors for rasterization and lighting.
    - **WebGPUBackend**: Hardware-accelerated pipeline utilizing a delegated architecture with specialized registries for resources, bindings, and frame execution.
    - **WebGLBackend**: Modern V1 implementation ensuring broad device compatibility.
- **Advanced Rendering Pipeline**:
    - Physically Based Rendering (PBR) with full IBL (Image-Based Lighting) support.
    - Real-time Shadowing system featuring PCSS and Cascaded Shadow Maps (CSM).
    - Post-Processing Suite: SSR, SSAO, TAA, FXAA, **Motion Blur**, **Depth of Field (DoF)**, and **Bloom** (HDR with soft-knee thresholding).
    - Advanced Volumetric Lighting featuring **ReSTIR** (Reservoir Spatiotemporal Importance Resampling).
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
- **`renderers/`**: Multi-backend implementations. Contains specialized logic for Software, WebGL, and WebGPU pipelines, organized into subdirectories for delegated execution.
- **`pipeline/`**: The execution frame graph, responsible for feature resolution, frame planning, and draw call preparation.
- **`simulation/`**: Unified simulation layer for animation, physics, and particles.
- **`foundation/`**: Core primitives (Color, IdGenerator, Logger, Platform) used engine-wide.
- **`workers/`**: Parallel processing infrastructure and `WorkerScheduler` for multi-threaded tasks.
- **`shaders/`**: Centralized shader repository. Includes a powerful **Rule-Based Shader Runtime** (`runtime/`) and backend-specific shader modules (`software/`, `webgpu/`, `webgl/`).
- **`animation/`, `physics/`, `particles/`**: Data structures and high-level systems for specialty simulations.
- **`maths/`**: A optimized math library focused on performance and memory efficiency.
- **`loaders/`**: Resource acquisition logic, featuring a robust glTF 2.0 parser.
- **`materials/` & `meshes/`**: Assets management for PBR shading and geometric data representation.
- **`cameras/` & `lights/`**: Specialized scene objects for view control and illumination.
- **`addons/`**: Extensibility folder for community and experimental features.

### `tests/` - Testing Suite
- Categorized automated tests covering lighting accuracy, physics consistency, animation state blending, and backend-specific feature verification. Headless Bun tests live under `tests/static/`, real browser-runtime tests live under `tests/browser/`, and benchmarks live under `tests/benchmarks/`.

## 🏗️ Architecture Overview

IgnisRenderer utilizes a data-driven design pattern to ensure scalability across diverse rendering targets and simulation complexities.

### 1. Data-Oriented Design (ECS)
The engine has transitioned to an **ECS-first architecture**. While traditional scene graph traversal is supported via the `Node` facade, the internal state is managed by the `ECSWorld`. This allows for high-efficiency queries and batch updates, particularly beneficial for complex simulations like particles and physics.

### 2. Multi-Stage Execution Pipeline
Each frame undergoes a rigid sequence of operations:
- **Feature Resolution**: The `FeatureResolver` inspects the scene to determine active requirements.
- **Warmup**: A robust pre-compilation phase (`WarmupPlanner`) prepares all necessary pipelines and resources before rendering.
- **Sync In**: Transfers manual `Node` changes into the `ECSWorld`.
- **Simulation Flow**: Animation, physics, and particle stages update their respective states in a unified timeline.
- **Prepared Scene Construction**: Values are collected into optimized draw packets, decoupling the scene graph from backend dispatches.
- **Backend-Agnostic Dispatch**: Commands are encoded through a unified interface, allowing the engine to switch between CPU and GPU rendering seamlessly.
- **Sync Out**: Propagates simulation results from ECS back to the `Node` facade.

### 3. Backend Strategy
- **Software Rasterizer**: Employs advanced CPU-side algorithms for sub-pixel accuracy and PBR shading. Multi-threading is enabled via the `WorkerScheduler`.
- **Modern Hardware Backends**: The WebGPU implementation leverages a state-of-the-art post-processing graph and a delegated resource management system.

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
