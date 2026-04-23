# AGENTS.md

This file provides critical context and collaboration guidance for AI/code agents working in the IgnisRenderer repository.

## Scope

- **IgnisRenderer** is a high-performance 3D rendering engine built in TypeScript.
- **Rendering Backends**:
	- **SoftwareBackend**: Multi-threaded CPU rasterizer with modular executors for rasterization, light evaluation, and post-processing.
	- **WebGPUBackend**: Hardware-accelerated pipeline utilizing a delegated architecture with specialized registries for resources, bindings, and frame execution.
	- **WebGLBackend**: Modernizing with a new V1 implementation for broad compatibility.
- **Core Architecture**: Entity Component System (ECS) backing a modular Scene Graph. `Node` acts as a high-level interface synchronized with the ECS. Integrated with Animation, Physics, and Particle simulation stages.

## Build & Test Commands

### Preferred Tooling
- **Runtimes & Package Managers**: MUST prioritize `bun`. Fall back to `node` and `npm` only if `bun` is unavailable.
- **Search Utilities**: MUST prioritize `rg` (ripgrep). Fall back to `grep` only if `rg` is unavailable.

### Commands
- **Dev server**: `bun run dev`
- **Build**: `bun run build`
- **Global Type Check**: `bunx tsc --noEmit`
- **Run all tests**: `bun run test`
- **Run single test**: `bun tests/<file>.mjs`

### Specialized Test Suites
- **Lighting**: `bun run test:lighting`, `bun run test:pointspot`, `bun run test:sh`
- **Geometry**: `bun run test:winding`, `bun run test:sparse`
- **Animation**: `bun tests/test_animation_core.mjs`, `bun tests/test_animation_state_blendtree.mjs`
- **Physics**: `bun tests/test_physics_system_bindings.mjs`, `bun tests/test_physics_adapter_contract.mjs`
- **WebGPU**: `bun tests/test_webgpu_bridge.mjs`, `bun tests/test_webgpu_post_graph.mjs`
- **WebGL**: `bun tests/test_webgl_backend_v2.mjs`, `bun tests/test_webgl_backend_stub.mjs`

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

### Docs Writing Guidelines (`docs/`)
- **Scope**: These rules apply only to `docs/*.md`. They do not apply to `README*.md` unless explicitly requested.
- **Language**: Use English by default. Keep technical terms in their canonical form and avoid mixed Chinese-English sentence structure.
- **Template (Strict Order)**: Every new or updated document in `docs/` must follow this section order:
	1. `# Title`
	2. `## Scope`
	3. `## Background` (use `N/A` if not applicable)
	4. `## API/Contract` (list input/output/constraints as contracts)
	5. `## Usage` (include at least one executable or verifiable example)
	6. `## Errors & Diagnostics` (list common errors and trigger conditions)
	7. `## Compatibility / Breaking Changes` (use `N/A` if not applicable)
- **Normative Tone**: Use RFC-style wording with `must`, `should`, and `may` for requirements and recommendations.
- **Identifier Formatting**: Wrap contract names, types, function names, and parameter names in backticks.
- **Precision**: Avoid ambiguous wording (for example, "usually", "maybe"). Use testable, verifiable statements.
- **Example Rules**:
	- Every code block must include a language tag (for example, `ts`, `wgsl`, `bash`).
	- Examples must match current API names and must not use removed interfaces.
- **Maintenance Rules**:
	- Any PR that changes public interfaces or behavior must update corresponding `docs/` files in the same change.
	- Migration documents must explicitly include breaking impact and replacement paths.

### Entity Component System (ECS) & Scene Graph
- **ECSWorld**: The core data structure managing Entities and their Components (`LocalTransform`, `WorldTransform`, `NodeRef`, `Visibility`, etc.).
- **Query System**: Use `world.query(["CompA", "CompB"])` for efficient entity filtering.
- **Node**: A high-level scene graph object that **synchronizes bi-directionally** with ECS entities. It provides a familiar API for translation, rotation, and scale while the ECS handles simulation state.
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
- **WebGPU Post-Processing Graph**: Modular plugin system supporting:
    - **SSAO**: Screen-Space Ambient Occlusion with depth-aware bilateral blur.
    - **TAA**: Temporal Anti-Aliasing with variance clamping and history rectification.
    - **SSR**: Screen-Space Reflections using **Hi-Z (Hierarchical Z-Buffer)** tracing.
    - **Volumetric Lighting**: Featuring **ReSTIR (Reservoir Spatiotemporal Importance Resampling)** for high-quality light scattering.
    - **Bloom**: Advanced HDR bloom with thresholding and soft-knee curves.
    - **Motion Blur**: Velocity-based blur utilizing the `gMotionDepth` buffer with shutter scale and sample control.
    - **Depth of Field (DoF)**: Cinematic bokeh effect with focus distance, range, and chromatic aberration.
    - **FXAA**: Fast Approximate Anti-Aliasing for broad compatibility.
- **Warmup System**: Robust pre-compilation phase (`WarmupPlanner`) to ensure all necessary pipelines and resources are prepared before rendering based on scene features.
- **Pipeline Stages**:
	1. **Feature Resolution**: Detects requirements (Shadows, IBL, Post-processing).
	2. **Warmup**: Pre-compiles shaders and pipelines if needed.
	3. **Sync In**: Syncs `Node` state to ECS.
	4. **Simulation**: Animation, Physics.
	5. **Transform Update**: Updates world matrices for the scene.
	6. **Prepared Scene Building**: Collects draw packets indexed by `MeshInstance`.
	7. **Backend Dispatch**: Software rasterization, GPU command encoding, or WebGL batching.
	8. **Sync Out**: Syncs ECS results back to `Node`.

### Shader Management
- **Avoid Inlining**: Do not embed shader code as long strings within TypeScript files. Use separate `.wgsl` or `.glsl` files.
- **Shader Runtime (`src/shaders/runtime/`)**: Advanced rule-based shader transformation system. Supports custom rules, validation, injection, and source mapping. 
- **Reorganized Folders**: Shader files are organized by backend applicability: `src/shaders/software/`, `src/shaders/webgpu/`, `src/shaders/webgl/`.

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
- **Resource Management**: Use explicit `.destroy()` methods. Managed through specialized backend registries (e.g., `WebGPUTextureRegistry`) with `FinalizationRegistry` as a safety net.

## Collaboration Workflow
1. Maintain backend-agnostic contracts in `src/core/` and `src/pipeline/`.
2. Ensure new features are accompanied by regression tests in `tests/`.
3. **Refactoring Policy**: Avoid large-scale refactorings unless explicitly requested. Prioritize stability.
4. When changing public APIs/behavior, update relevant `docs/` first, then add or update tests in the same PR.
5. Update `AGENTS.md` if core architectural patterns change.
