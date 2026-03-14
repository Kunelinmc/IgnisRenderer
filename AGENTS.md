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

## Architecture & Conventions

### Entity Component System (ECS) & Scene Graph
- **ECSWorld**: The core data structure managing Entities and their Components (`LocalTransform`, `WorldTransform`, `NodeRef`, `Visibility`, etc.).
- **Query System**: Use `world.query(["CompA", "CompB"])` for efficient entity filtering.
- **Node**: Now a **deprecated compatibility facade** over ECS entities. Represents translation, rotation, and scale. Updates are synced between ECS components and the `Node` facade via `ECSWorld.syncNodeToEntity`.
- **Simulation Logic**: Integrated into the rendering pipeline or updated per frame:
	- `AnimationSimulationStage`: Handles mixers, blend trees, and skeletal updates.
	- `PhysicsSystem`: Manages collision detection, vehicle physics, and syncing rigidbodies with external adapters (Rapier3D, Ammo.js).
	- `ParticleSimulationStage`: Updates particle state before rendering.

### Advanced Rendering Features
- **WebGPU Post-Processing Graph**: Supports SSAO, SSR, TAA, FXAA, and Volumetric Lighting.
- **Pipeline Stages**:
	1. **Feature Resolution**: Detects requirements (Shadows, IBL, Post-processing).
	2. **Simulation**: Animation, Particles, Physics.
	3. **Prepared Scene Building**: Collects draw packets indexed by `MeshInstance` and `MeshAsset`.
	4. **Backend Dispatch**: Software rasterization, GPU command encoding, or WebGL batching.

### Mathematics
- `src/maths/` is the single source of truth for math behavior.
- **Matrix Rule**: `A.multiply(B)` performs `A = A * B` (in-place).
- **Coordinate Space**: Right-handed (Y-up, Z-towards viewer).

### Time Units
- **ALL simulation logic MUST use seconds**.
- Variables should be suffixed with `Seconds` (e.g., `deltaTimeSeconds`).
- Convert from `ms` to `seconds` at the entry points (e.g., in `Renderer` loops).

## Implementation Guidelines

- **Zero-Allocation Loops**: Especially critical in `SoftwareBackend` hot paths (Rasterizer). Use pre-allocated math objects and avoided `new` inside loops.
- **Adapter Pattern**: Physics and Animation systems use adapters/plugins to remain engine-agnostic where possible.
- **Fail Fast**: Throw descriptive errors for invalid scene hierarchies or backend configurations.
- **WebGPU Safety**: Always check device/pipeline status before submitting commands.

## Collaboration Workflow
1. Maintain backend-agnostic contracts in `src/core/` and `src/pipeline/`.
2. Ensure new features are accompanied by regression tests in `tests/`.
3. Update `AGENTS.md` if core architectural patterns change.
