# AGENTS.md

This file provides critical context and collaboration guidance for AI/code
agents working in the IgnisRenderer repository.

## Scope

- **IgnisRenderer** is a high-performance 3D rendering engine built in
  TypeScript.
- **Rendering Backends**:
  - **SoftwareBackend**: CPU rasterizer pipeline.
  - **WebGPUBackend**: hardware-accelerated WebGPU pipeline.
  - **WebGLBackend**: experimental/stub legacy backend.
- **Current Core Architecture**: Scene Graph (`Node`) +
  `MeshAsset`/`MeshInstance`.

## Build & Test Commands

- **Dev server**: `npm run dev`
- **Run all tests**: `npm test`
- **Run single test**: `npx tsx tests/<file>.mjs`
- **Available suites**:
  - `npm run test:lighting`
  - `npm run test:pointspot`
  - `npm run test:sh`
  - `npm run test:winding`
  - `npm run test:sparse`

## Code Style Guidelines

### Imports & Modules

- **Source files (`src/`)**: use extensionless relative imports.
  - Correct: `import { Vector3 } from '../maths/Vector3'`
  - Incorrect: `import { Vector3 } from '../maths/Vector3.ts'`
- **Test files (`tests/`)**: use `.mjs` or `.ts` extensions.
  - Correct: `import { Light } from '../src/lights/Light.ts'`
- Use `import type { ... }` for type-only dependencies.
- Import grouping: external first, then internal (with a blank line).

### Formatting

- Indentation: tabs (size 4)
- Semicolons: omit unless syntactically necessary
- Strings: single quotes unless interpolation is required
- Line length: target 80-100 characters

### Naming

- Classes/Interfaces/Types: `PascalCase`
- Methods/Variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: prefix `_`
- Files: `PascalCase.ts` for classes, `camelCase.ts` for utility modules

## Architecture & Conventions

### Scene Graph Model (Breaking)

- `Node` is the base transform unit:
  - `id/name/parent/children/visible`
  - `position/quaternion/scale`
  - `localMatrix/worldMatrix`
- `MeshAsset`: shared geometry/material resource container (primitives +
  local bounds).
- `MeshInstance extends Node`: scene node referencing one `MeshAsset`.
- `Light`, `Camera`, and `ParticleSystem` all extend `Node`.
- `DirectionalLight` and `SpotLight` keep `direction` as a **local-space**
  vector.

### Scene API

- Use generic graph operations:
  - `scene.add(node)`
  - `scene.remove(node)`
  - `scene.traverse(visitor)`
  - `scene.contains(node)`
- Use typed queries when needed:
  - `scene.getMeshInstances()`
  - `scene.getLights()`
  - `scene.getCameras()`
  - `scene.getParticleSystems()`
- Do not reintroduce split entry points like `addModel/addLight/...`.

### Renderer Rules

- Renderer keeps active `camera`, but camera must belong to the active scene
  graph.
- `setScene` and `setCamera` validate camera membership and throw on invalid
  usage.
- World transforms are updated per frame from scene graph traversal
  (`scene.updateWorldMatrices()`).

### Loader Rules

- `GLTFLoader`/`GLBLoader` return a `Node` root (not `Model`).
- Preserve glTF hierarchy; do not bake node transforms into vertex data.
- Parse glTF cameras and `KHR_lights_punctual` into scene nodes.
- For multi-attachment glTF nodes (mesh/light/camera), use container nodes.
- `OBJLoader` also returns a `Node` root with `MeshInstance` children.

### Rendering Pipeline

1. `Renderer` orchestrates frame lifecycle and feature toggles.
2. `IRenderBackend` abstracts backend APIs.
3. `FramePlanner` builds pass sequences.
4. `PreparedSceneBuilder` traverses `MeshInstance` data for draw packets.
5. Shader implementations:
   - Software: `src/shaders/software/`
   - WebGPU: `src/shaders/webgpu/`

### Coordinate Spaces

- World space: right-handed (X right, Y up, Z toward viewer)
- View space: camera-relative, forward is -Z
- Clip space: homogeneous `(x, y, z, w)`
- NDC:
  - Software: Z in `[-1, 1]`
  - WebGPU: Z in `[0, 1]`
- Screen space: origin at top-left

### Performance Patterns

- Software backend: avoid allocations in hot loops.
- WebGPU backend: minimize state churn, cache pipelines/bind groups.
- Use typed arrays for large data paths (`Float32Array`, `Uint32Array`).

## Implementation Guidelines

### Mathematical Truth

- `src/maths/` is source-of-truth for math behavior.
- Matrix multiply rule: `A.multiply(B)` means `A = A * B`.
- For normals with non-uniform scale, use `Matrix4.normalMatrix()`.

### Error Handling

- Fail fast with explicit `Error` on invalid setup/input.
- Keep WebGPU validation checks in async-safe paths.

## Collaboration Workflow

1. Keep backend-agnostic contracts intact unless task explicitly needs changes.
2. Make minimal, targeted changes for the requested scope.
3. Document public API changes with brief rationale.
4. Run `npm test` after renderer/math/pipeline changes.
