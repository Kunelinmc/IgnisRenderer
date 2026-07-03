# AGENTS.md

This file provides the high-priority working rules for AI/code agents in the
IgnisRenderer repository. Keep this file loaded for every task. Load the linked
reference documents when a task touches the relevant subsystem.

## Repository Scope

- IgnisRenderer is a high-performance 3D rendering engine built in TypeScript.
- Core contracts must stay backend-agnostic in `src/core/`, `src/pipeline/`,
  and `src/postprocess/`.
- Rendering backends are one-shot renderer runtimes. A backend instance must
  attach to at most one `Renderer`; create a new backend instance for another
  renderer.
- `Renderer` owns the public facade. Backends own device lifecycle, frame
  execution, backend resources, and backend-specific passes.

## Required Reference Docs

These are contributor-facing project documents that `AGENTS.md` links as
required references. Read them before changing related behavior:

- `docs/architecture/engine-architecture.md`: backend roles, ECS and scene graph
  synchronization, simulation ownership, foundation utilities, and workers.
- `docs/rendering/rendering-pipeline-and-shader-contracts.md`: frame pipeline
  stages, coordinate system, matrix conventions, color space, shader layout,
  vertex attributes, post-processing, and resource/performance constraints.
- `docs/renderer-contract.md`: renderer/backend lifecycle, frame scheduling,
  device lifecycle, extension registry, and render target contracts.
- `docs/postprocess/backend-execution-contract.md` and
  `docs/postprocess/cross-backend-contract.md`: post-process runtime ownership
  and cross-backend pass behavior.
- `docs/workflow/commit-message-guidelines.md`: required commit header format,
  approved types, body/footer rules, and examples. Read before committing.
- Backend-specific contract files in `docs/` when changing WebGPU, WebGL,
  Software, warmup, lighting, shadows, materials, or loader behavior.

## Build & Test Commands

### Preferred Tooling

- Prioritize `bun`. Fall back to `node` and `npm` only if `bun` is unavailable.
- Prioritize `rg` for file and text search. Fall back only if `rg` is
  unavailable.

### Commands

- Dev server: `bun run dev`
- Build: `bun run build`
- Global type check: `bunx tsc --noEmit`
- Run all static tests: `bun run test`
- Run browser tests: `bun run test:browser`
- Run one static test: `bun tests/static/<subsystem>/<file>.mjs`

### Specialized Test Suites

- Lighting: `bun run test:lighting`, `bun run test:pointspot`,
  `bun run test:sh`
- Geometry: `bun run test:winding`, `bun run test:sparse`
- Animation: `bun tests/static/animation/test_animation_core.mjs`,
  `bun tests/static/animation/test_animation_state_blendtree.mjs`
- Physics: `bun tests/static/physics/test_physics_system_bindings.mjs`,
  `bun tests/static/physics/test_physics_adapter_contract.mjs`
- WebGPU: `bun tests/static/webgpu/test_webgpu_bridge.mjs`,
  `bun tests/static/webgpu/test_webgpu_post_graph.mjs`
- WebGL: `bun tests/static/webgl/test_webgl_backend_v2.mjs`,
  `bun tests/static/webgl/test_webgl_backend_stub.mjs`

## Code Style Guidelines

### Imports & Modules

- Source files in `src/` must use extensionless relative imports, for example
  `import { Node } from "../core/Node"`.
- Test files in `tests/` must use `.mjs` or `.ts` extensions.
- Use `import type { ... }` for type-only dependencies.
- Group external imports before internal imports with one blank line between
  groups.

### Formatting

- Use tabs for indentation.
- Terminate statements with semicolons.
- Use double quotes for string literals. Use single quotes only for nested
  quotes or standard-required syntax.
- Target 80-100 characters per line.

### Naming

- Use `PascalCase` for classes, interfaces, and types.
- Use `camelCase` for methods, variables, and functions.
- Use `UPPER_SNAKE_CASE` for constants.
- Prefix private/internal members with `_`.
- Use `PascalCase.ts` for files containing classes.
- Use `camelCase.ts` for utility and logic modules.

### Documentation

- Public methods and properties should have JSDoc when their behavior,
  constraints, or side effects are not obvious from the name and type.
- Simple variables, simple properties, and methods that only perform an
  obvious action or return no value may omit the entire JSDoc when the
  signature is self-explanatory.
- New externally exposed public methods must document purpose and any relevant
  parameters, return value, constraints, and observable side effects. Omit
  sections that do not apply, such as return value documentation for `void`
  methods.
- Public methods or interfaces that exist only for TypeScript contracts,
  backend bridges, tests, or renderer orchestration must include `@internal`.
  The JSDoc should name the owning subsystem and preferred public alternative.
- Add concise inline comments for non-obvious behavior such as matrix math,
  shader packing, or lifecycle ordering.

### Docs Writing Guidelines

- These rules apply to `docs/**/*.md`. They do not apply to `README*.md` unless
  explicitly requested.
- Use English by default. Keep technical terms in their canonical form.
- Every new or updated document in `docs/` must follow this section order:
  1. `# Title`
  2. `## Scope`
  3. `## Background`
  4. `## API/Contract`
  5. `## Usage`
  6. `## Errors & Diagnostics`
  7. `## Compatibility / Breaking Changes`
- Use RFC-style wording with `must`, `should`, and `may`.
- Wrap contract names, types, functions, and parameters in backticks.
- Avoid ambiguous wording. Prefer testable statements.
- Every code block must include a language tag.
- Examples must match current API names and must not use removed interfaces.
- Public interface or behavior changes must update corresponding `docs/` files
  in the same change.

## Architecture Rules

- Maintain separation between definition layers, such as interfaces and types,
  and logic layers, such as systems and simulation stages.
- Do not mix backend-specific resource ownership into backend-agnostic public
  contracts.
- `src/foundation/Error.ts` is the central definition file for custom error
  subclasses. New custom `Error` subclasses must be defined there and imported
  by owning subsystems.
- Ordinary `throw new Error(...)` usage does not require a centralized
  definition.
- Simulation logic must use seconds. Use the `deltaTimeSeconds` suffix for time
  step variables.
- Hot paths must avoid avoidable allocation. Use pre-allocated math objects
  where practical.
- Resources that own native or backend state must expose explicit `destroy()`
  methods.

## Shader & Rendering Rules

- Do not embed shader code as long strings inside TypeScript files. Use separate
  `.wgsl` or `.glsl` files.
- Shader files must stay organized by backend applicability under
  `src/shaders/software/`, `src/shaders/webgpu/`, and `src/shaders/webgl/`.
- Post-processing is a backend-owned `"postprocess"` backend pass. Do not add
  public backend post-process graph registration APIs or hardcode pass kernels
  outside `src/postprocess/passes/`.
- WebGPU deferred lighting is internal to the WebGPU `main-opaque` pass and
  must not add global renderer frame-pass stages for Software or WebGL.
- `IBLPrefilter` owns CPU, worker, and WebGPU environment specular
  prefiltering. `Renderer` must not schedule environment IBL bake/update work
  or expose environment IBL update APIs.

## Collaboration Workflow

1. Keep changes scoped to the requested subsystem and existing architecture.
2. Avoid large-scale refactoring unless explicitly requested.
3. Add or update regression tests for new behavior and bug fixes.
4. When changing public APIs or behavior, update relevant `docs/` first, then
   update tests in the same change.
5. Update `AGENTS.md` only when high-priority agent guidance changes.
6. When architecture changes, audit stale descriptions in `AGENTS.md`, `docs/`,
   `README*.md`, and relevant code comments.

## Commit Message Guidelines

- Commit headers must use Conventional Commits: `type(scope)!: subject`.
- Use the smallest accurate approved type.
- Use `!` and `BREAKING CHANGE:` for breaking API, behavior, file layout,
  shader contract, or backend contract changes.
- Read `docs/workflow/commit-message-guidelines.md` before creating commits.
