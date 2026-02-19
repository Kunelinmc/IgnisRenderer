# AGENTS.md

This file provides critical context and collaboration guidance for AI/code agents working in the IgnisRenderer repository.

## Scope

- Applies to the entire repository. IgnisRenderer is a high-performance 3D software rendering engine built from scratch in TypeScript, implementing a complete graphics pipeline on the CPU.

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
  - *Correct*: `import { Vector3 } from '../maths/Vector3'`
  - *Incorrect*: `import { Vector3 } from '../maths/Vector3.ts'`
- **Test Files (`tests/`)**: Must use `.ts` extensions for `tsx` compatibility.
  - *Correct*: `import { Light } from '../src/lights/Light.ts'`
- **Type-only imports**: Always prefer `import type { ... }` for interface-only dependencies.
- **Grouping**: External libraries first, then internal modules, separated by a blank line.

### Formatting
- **Indentation**: Use **tabs** (standard size 4).
- **Semicolons**: **Omit** semicolons (ASI-friendly). Use them only when syntactically necessary (e.g., before an array literal starting a line).
- **Strings**: Use **single quotes** (`'`) unless interpolation (`${}`) is required.
- **Line Length**: Aim for 80-100 characters for readability.

### Naming Conventions
- **Classes/Interfaces/Types**: `PascalCase` (e.g., `Matrix4`, `PBRMaterial`). Interfaces should NOT be prefixed with `I` unless they are truly generic (e.g., `IVector3` is allowed, but prefer `Camera`).
- **Methods/Variables**: `camelCase` (e.g., `computeLight`, `intensity`).
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_SH_ORDER`).
- **Private Members**: Prefix with an underscore `_` (e.g., `private _intensity`).
- **Files**: `PascalCase.ts` for classes; `camelCase.ts` for utilities and types.

### Types & Interfaces
- **Safety**: `strict: false` is set for migration, but agents must avoid `any`. Use `unknown` or generics. Never use double assertions (`as unknown as`).
- **Object Literals**: Prefer explicit types for return objects to ensure property consistency.

## Architecture & Conventions

### Rendering Pipeline Flow
1. **Scene Update**: Light matrices and Spherical Harmonics (SH) are updated.
2. **Shadow Pass**: `ShadowRenderer` generates depth maps for shadow-casting lights.
3. **Reflection Pass**: `ReflectionRenderer` renders the scene from reflection planes.
4. **Main Pass**: 
   - `Projector` transforms vertices to screen space.
   - Faces are culled (backface) and sorted by depth (for transparency).
   - `Rasterizer` fills triangles and computes per-pixel lighting.
5. **Post-Processing**: `PostProcessor` applies FXAA, Volumetric Lighting, and Gamma correction.

### Coordinate Systems & Space Transitions
- **World Space**: Right-handed (X: Right, Y: Up, Z: Towards Viewer).
- **View Space**: Camera-relative. Eye at origin, -Z is the forward looking direction.
- **Clip Space**: Homogeneous coordinates (x, y, z, w). W is used for perspective division.
- **NDC (Normalized Device Coordinates)**: After perspective divide. range [-1, 1].
- **Screen Space**: (0,0) at top-left, pixel centers at offset +0.5.
- **Depth Buffer**: Stores **linear camera-space depth** (positive distance from camera plane).

### Matrix & Math
- **Order**: Row-major conceptual storage: `elements[row][col]`.
- **Transformation**: `lookAt` creates a system where -Z is forward.
- **Normal Matrix**: Use the transpose of the inverse of the top-left 3x3 of the model matrix.
- **Constants**: Use `src/core/Constants.ts` or local static constants. Avoid magic numbers.

## Implementation Guidelines

### Performance Patterns
- **Inner Loops**: Avoid object allocation (e.g., `new Vector3()`) inside `drawTriangle` or `computeContribution`. Use in-place operations or scratchpad variables.
- **Early Returns**: Validate conditions early to minimize nesting.
- **Pre-allocation**: Use typed arrays (e.g., `Float32Array`) for large data buffers.
- **Bitwise Ops**: Use `| 0` for fast integer truncation when calculating pixel indices.

### Error Handling
- **Fail Fast**: Throw explicit `Error` objects on invalid input.
- **No Silencing**: Never swallow errors. If a fallback is necessary, log a warning.

## Testing & Verification

### Testing Policy
- **Regression**: NEVER modify existing tests to "fix" a build unless the behavior change is intentional.
- **Framework**: Uses Node.js `node:assert/strict`. Test files are `.mjs`.
- **Verification**: Always run the relevant test suite after modifying math or core rendering logic.

### Common Gotchas
- **Matrix Multiplication**: `A.multiply(B)` is often `A = A * B`. Check if a static `multiply(A, B, out)` is available for non-destructive ops.
- **Normal Transformation**: Always use `Matrix4.normalMatrix()` to transform normals; do not use the raw model matrix if scaling is involved.
- **Alpha Blending**: Ensure `transparentFaces` are sorted back-to-front before rasterization.
- **Backface Culling**: Dependent on winding order (CCW by default).

## Glossary of Key Terms

- **ProjectedFace**: A triangle that has been transformed to screen space, containing depth and normal data.
- **SH (Spherical Harmonics)**: Low-frequency environmental lighting representation used for ambient and diffuse IBL.
- **Rasterizer**: The core loop that iterates over pixels inside a triangle and computes the final color.
- **Fragment**: A potential pixel on screen. In this engine, fragment shading is done during rasterization.
- **W-Divide**: The division of x, y, z by w to transition from Clip Space to NDC.

## Design Philosophy

- **Explicit over Implicit**: Math operations should be clear. Prefer named methods over operator overloading.
- **Composition over Inheritance**: Rendering features should be modular processors rather than deep class hierarchies.
- **Performance First**: Since this is a CPU renderer, algorithmic efficiency and minimizing allocations are paramount.
- **Safety with Flexibility**: TypeScript is used for structure, but low-level array manipulation is allowed for speed.

## Collaboration Workflow
- **Minimal Changes**: Avoid large-scale refactors. Focus on the requested task.
- **Documentation**: Use JSDoc for public members. Focus on "why" (intent) over "what".
- **Ambiguity**: If math conventions or coordinate spaces are unclear, ask a focused question before proceeding.
