# AGENTS.md

This file provides critical context and collaboration guidance for AI/code agents working in the IgnisRenderer repository.

## Scope

- **IgnisRenderer** is a high-performance 3D rendering engine built in TypeScript.
- **Architectural Shift**: Originally a pure CPU software renderer, it now supports multiple backends:
  - **SoftwareBackend**: A complete C-style graphics pipeline running on the CPU (Rasterizer-based).
  - **WebGPUBackend**: A modern hardware-accelerated pipeline using WebGPU (WGSL/Compute/Raster).
  - **WebGLBackend**: (Experimental) A legacy hardware backend.

## Build & Test Commands

- **Dev server**: `npm run dev` - Starts the Vite development server for real-time preview.
- **Run all tests**: `npm test` - Executes the full regression suite via `tests/run_all.mjs`.
- **Run single test**: `npx tsx tests/<file>.mjs` (e.g., `npx tsx tests/test_lighting.mjs`).
- **Available test suites**:
  - `npm run test:lighting` - Core lighting calculations and contributions.
  - `npm run test:pointspot` - Point and Spot light attenuation and cones.
  - `npm run test:sh` - Spherical Harmonics projection and reconstruction.
  - `npm run test:winding` - Geometry winding order and normal consistency.
  - `npm run test:sparse` - glTF sparse accessor handling.

## Code Style Guidelines

### Imports & Modules

- **Source Files (`src/`)**: Use extensionless relative imports.
  - _Correct_: `import { Vector3 } from '../maths/Vector3'`
  - _Incorrect_: `import { Vector3 } from '../maths/Vector3.ts'`
- **Test Files (`tests/`)**: Must use `.mjs` or `.ts` extensions.
  - _Correct_: `import { Light } from '../src/lights/Light.ts'`
- **Type-only imports**: Always prefer `import type { ... }` for interface-only dependencies.
- **Grouping**: External libraries first, then internal modules, separated by a blank line.

### Formatting

- **Indentation**: Use **tabs** (standard size 4).
- **Semicolons**: **Omit** semicolons (ASI-friendly). Use them only when syntactically necessary.
- **Strings**: Use **single quotes** (`'`) unless interpolation (`${}`) is required.
- **Line Length**: Aim for 80-100 characters for readability.

### Naming Conventions

- **Classes/Interfaces/Types**: `PascalCase` (e.g., `Matrix4`, `PBRMaterial`).
- **Methods/Variables**: `camelCase` (e.g., `computeLight`, `intensity`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SH_ORDER`).
- **Private Members**: Prefix with an underscore `_` (e.g., `private _intensity`).
- **Files**: `PascalCase.ts` for classes; `camelCase.ts` for utilities and types.

## Architecture & Conventions

### Rendering Architecture

1.  **Renderer**: The central orchestrator (`src/renderers/Renderer.ts`). it manages the scene, camera, and features.
2.  **IRenderBackend**: An abstraction layer for different rendering APIs (`src/renderers/IRenderBackend.ts`).
3.  **FramePlanner**: Analyzes scene requirements and active features to build a sequence of `FramePass` objects.
4.  **PreparedSceneBuilder**: Pre-processes the scene into a backend-agnostic layout.
5.  **Shaders**:
    - **Software**: TypeScript functions in `src/shaders/software/`.
    - **WebGPU**: WGSL source code in `src/shaders/webgpu/`.

### Coordinate Systems & Space Transitions

- **World Space**: Right-handed (X: Right, Y: Up, Z: Towards Viewer).
- **View Space**: Camera-relative. Eye at origin, -Z is the forward looking direction.
- **Clip Space**: Homogeneous coordinates (x, y, z, w).
- **NDC (Normalized Device Coordinates)**:
  - **Software**: Z range [-1, 1].
  - **WebGPU**: Z range [0, 1]. The projection matrices in `Matrix4` typically assume [-1, 1], so WebGPU shaders or drivers must handle the remap.
- **Screen Space**: (0,0) at top-left.

### Performance Patterns

- **Software Backend**: Avoid object allocation (e.g., `new Vector3()`) in hot loops (`drawTriangle`). Use in-place operations.
- **WebGPU Backend**: Minimize state changes. Cache BindGroups and Pipelines. Use `GPUDevice.pushErrorScope` for validation.
- **Memory**: Use `Float32Array` or `Uint32Array` for large buffers (Vertex, Index, Framebuffer).

## Implementation Guidelines

### Mathematical Truth

- The math classes in `src/maths/` are the **Source of Truth**.
- **Matrix Multi Order**: `A.multiply(B)` results in `A = A * B`.
- **Normal Matrix**: Always use `Matrix4.normalMatrix()` (transpose of inverse) for normals, especially if non-uniform scaling is present.

### Error Handling

- **Fail Fast**: Throw explicit `Error` objects on invalid input during initialization.
- **Validation**: WebGPU backend should use async validation checks.

## Collaboration Workflow

1.  **Consistency**: Follow the existing backend-agnostic patterns if possible.
2.  **Minimal Changes**: Focus on the requested task; avoid unsolicited refactors unless asked.
3.  **Documentation**: Use JSDoc for public API members. Explain the "Why" behind math implementations.
4.  **Verification**: Always run `npm test` after modifying core rendering or math logic.
