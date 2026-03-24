# AGENTS.md

This file provides critical context and collaboration guidance for AI/code agents working in the IgnisRenderer repository.

## Scope

- **IgnisRenderer** is a high-performance 3D rendering engine built in TypeScript.
- **Rendering Backends**:
	- **SoftwareBackend**: Multi-threaded CPU rasterizer pipeline with custom PBR shading.
	- **WebGPUBackend**: Hardware-accelerated pipeline with advanced post-processing features.
	- **WebGLBackend**: Modernizing with a new V1 implementation for broad compatibility.
- **Core Architecture**: Entity Component System (ECS) backing a modular Scene Graph. `Node` acts as a compatibility facade. Integrated with Animation, Physics, and Particle simulation stages.

## Build & Test Commands

- **Dev server**: `npm run dev`
- **Build**: `npm run build`
- **Run all tests**: `npm test`
- **Run single test**: `npx tsx tests/<file>.mjs`

### Specialized Test Suites
- **Lighting**: `npm run test:lighting`, `npm run test:pointspot`, `npm run test:sh`
- **Geometry**: `npm run test:winding`, `npm run test:sparse`
- **Animation**: `npx tsx tests/test_animation_core.mjs`, `npx tsx tests/test_animation_state_blendtree.mjs`
- **Physics**: `npx tsx tests/test_physics_system_bindings.mjs`, `npx tsx tests/test_physics_adapter_contract.mjs`
- **WebGPU**: `npx tsx tests/test_webgpu_bridge.mjs`, `npx tsx tests/test_webgpu_post_graph.mjs`
- **WebGL**: `npx tsx tests/test_webgl_backend_v1.mjs`, `npx tsx tests/test_webgl_backend_stub.mjs`

## Code Style Guidelines

### Imports & Modules
- **Source files (`src/`)**: Use extensionless relative imports. (e.g., `import { Node } from "../core/Node"`)
- **Test files (`tests/`)**: Use `.mjs` or `.ts` extensions.
- Use `import type { ... }` for type-only dependencies to optimize build bundles.
- Import grouping: External dependencies first, then internal modules (separated by a blank line).

### Formatting
- **Indentation**: Tabs (size 4)
- **Semicolons**: Include for all statement terminations.
- **Strings**: Use double quotes for string literals. Single quotes are only for nested quotes or when required by standard usage.
- **Line length**: Target 80-100 characters.

### Naming
- **PascalCase**: Classes, Interfaces, Types.
- **camelCase**: Methods, Variables, Functions.
- **UPPER_SNAKE_CASE**: Constants.
- **_prefix**: Private/internal members (e.g., `_myPrivateVar`, `_internalMethod`).
- **PascalCase.ts**: Files containing classes.
- **camelCase.ts**: Files for utility/logic modules.

### Documentation
- **Comments & JSDoc**: Use JSDoc for all public methods and properties. Include clear inline comments for complex logic (e.g., matrix math, shader packing).

### Entity Component System (ECS) & Scene Graph
- **ECSWorld**: The core data structure managing Entities and their Components (`LocalTransform`, `WorldTransform`, `NodeRef`, `Visibility`, etc.).
- **Query System**: Use `world.query(["CompA", "CompB"])` for efficient entity filtering.
- **Node**: Now a **deprecated compatibility facade** over ECS entities. Represents translation, rotation, and scale.
- **Synchronization Flow**:
	- `Sync In` (`Scene.syncNodeToECS`): Transfers manual `Node` changes into ECS components.
	- `Sync Out` (`Scene.syncECSToNode`): Propagates ECS simulation results (Physics/Animation) back to the `Node` facade.
- **Architectural Integrity**: Maintain separation between **Definition layer** (Interfaces, Types) and **Logic layer** (Systems, Simulation Stages). Avoid mixing responsibilities within a single class.
- **Simulation Logic**: Integrated into the rendering pipeline via `src/simulation/` runtime modules:
	- `AnimationSimulationStage`: Handles mixers, blend trees, and skeletal updates.
	- `PhysicsSystem`: Manages collision detection and rigid body dynamics via adapters.
	- `ParticleSimulationStage`: Updates high-density particle state.

### Foundation & Utility Layer
- **`src/foundation/`**: Core primitives used across the entire engine.
	- `Color.ts`: HSL/RGB parsing and linear-space color utilities.
	- `IdGenerator.ts`: Deterministic ID generation for resources.
	- `Platform.ts`: Environment detection and browser-specific capability checks.
- **`src/workers/`**: Multi-threading infrastructure.
	- `WorkerScheduler`: Manages a pool of Web Workers for parallel tasks (e.g., SoftwareBackend rasterization).
	- `transports.ts`: Efficient zero-copy data transfer using `SharedArrayBuffer` where available.

### Advanced Rendering Features
- **WebGPU Post-Processing Graph**: Modular plugin system supporting SSAO, SSR, TAA, FXAA, and Volumetric Lighting.
- **Pipeline Stages**:
	1. **Feature Resolution**: Detects requirements (Shadows, IBL, Post-processing).
	2. **Sync In**: Syncs `Node` state to ECS.
	3. **Simulation**: Animation, Physics.
	4. **Transform Update**: Updates world matrices for the scene.
	5. **Prepared Scene Building**: Collects draw packets indexed by `MeshInstance`.
	6. **Backend Dispatch**: Software rasterization, GPU command encoding, or WebGL batching.
	7. **Sync Out**: Syncs ECS results back to `Node`.

### Shader Management
- **Avoid Inlining**: Do not embed shader code as long strings within TypeScript files. Use separate `.wgsl` or `.glsl` files.
- **Shader Runtime**: `src/shaders/runtime.ts` handles dynamic shader preprocessing, including `#include` resolution and feature-based permutation generation.

## Core Conventions

### Mathematics & Coordinate System
- **Handedness**: Right-Handed.
- **Axis Orientation**:
	- **+Y**: Up
	- **-Z**: Forward (Camera view direction)
	- **+X**: Right
- **Camera Logic**: Carefully handle differences between **Perspective** (FOV-based, non-linear depth) and **Orthographic** (volume-based, linear depth) projections.
- **Matrices**:
	- Internal representation (`src/maths/Matrix4.ts`): Row-major `number[row][col]`.
	- GPU Buffers (WGSL/GLSL): Column-major `Float32Array`.
	- Multiplicative Order: `A.multiply(B)` performs `A = A * B`.
- **Projection**: Internal matrices target standard NDC range `[-1, 1]` for Z.

### Color Space & Lighting
- **Internal Calculations**: All lighting and shading calculations are performed in **Linear space**.
- **Gamma Correction**: Assumes **Gamma 2.2** for encoding/decoding.
- **Texture Decoding**:
	- Shaders assume textures are encoded in **sRGB** by default and decode them to **Linear** during sampling.
	- Linear textures (e.g., normal maps, roughness) MUST be flagged to bypass decoding.

### Shader & Material Assumptions
- **PBR Model**: GGX (NDF), Smith-Schlick (Geometry), and Fresnel-Schlick.
- **Vertex Attributes**:
	- `shaderLocation 0`: Position (`vec3`)
	- `shaderLocation 1`: Normal (`vec3`)
	- `shaderLocation 2`: UV0 (`vec2`)
	- `shaderLocation 3`: UV1 (`vec2`)
	- `shaderLocation 4`: Tangent (`vec4`, where `w` is handedness)

### Performance & Memory
- **Time Units**: ALL simulation logic MUST use seconds. Use `deltaTimeSeconds` suffix.
- **Zero-Allocation Loops**: Use pre-allocated math objects (e.g., `_tempVec`) and avoid `new` inside hot paths.
- **Adapter Pattern**: Used for Physics to allow switching backend implementations.
- **Resource Management**: Use explicit `.destroy()` methods. A `FinalizationRegistry` (in `WebGPUBackend`) acts as a safety net for GPU resources.

## Collaboration Workflow
1. Maintain backend-agnostic contracts in `src/core/` and `src/pipeline/`.
2. Ensure new features are accompanied by regression tests in `tests/`.
3. **Refactoring Policy**: Avoid large-scale refactorings unless explicitly requested. Prioritize stability.
4. Update `AGENTS.md` if core architectural patterns change.
