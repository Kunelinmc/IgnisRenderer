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

- `docs/architecture/engine.md`: backend roles, ECS and
  scene graph synchronization, simulation ownership, foundation utilities,
  and workers.
- `docs/architecture/rendering.md`:
  frame pipeline stages, coordinate system, matrix conventions, color space,
  shader layout, vertex attributes, post-processing, and
  resource/performance constraints.
- `docs/contracts/renderer.md`: renderer/backend
  lifecycle, frame scheduling, device lifecycle, extension registry, and
  render target contracts.
- `docs/contracts/postprocess.md`:
  post-process runtime ownership and cross-backend pass behavior.
- `docs/contributing/README.md`: documentation workflow and required commit
  header
  format, approved types, body/footer rules, and examples. Read before
  committing.
- Relevant subsystem files under `docs/contracts/`, including
  `docs/contracts/webgpu.md` and `docs/contracts/webgl.md`, when changing
  backend, warmup, lighting, shadows, materials, or loader behavior.

## Build & Test Commands

### Preferred Tooling

- Prioritize `bun`. Fall back to `node` and `npm` only if `bun` is unavailable.
- Prioritize `rg` for file and text search. Fall back only if `rg` is
  unavailable.
- Avoid token-wasteful discovery patterns. Do not run unscoped `rg --files` from
  the repository root, read entire files in one pass, or use low-hit-rate broad
  searches when a narrower path, glob, symbol, or line-limited read can answer
  the question.

### Commands

- Dev server: `bun run dev`
- Build: `bun run build`
- Global type check: `bun run typecheck`
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
- WebGPU: `bun tests/run_all.mjs tests/static/webgpu`,
  `bun tests/static/webgpu/test_webgpu_post_graph.mjs`
- WebGL: `bun tests/run_all.mjs tests/static/webgl`,
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

- These rules apply to every Markdown file under `docs/`.
- Use `docs/public/` only for consumer-facing documentation based on exports
  from `src/index.ts`.
- Place normative implementation and backend contracts under `docs/contracts/`,
  architecture under `docs/architecture/`, maintainer references under
  `docs/reference/`, migrations under `docs/migrations/`, and repository
  workflows under `docs/contributing/`.
- Maintained Markdown paths must use `docs/<category>/<document>.md`. Do not add
  nested subsystem directories.
- Architecture documents explain responsibility, flow, and design boundaries.
  Normative lifecycle, fallback, diagnostic, and compatibility requirements
  must live in contracts.
- A single effect, capability, probe behavior, loader variant, material
  property, shader option, or backend pass must be a section of its owning
  subsystem document rather than a standalone document.
- Do not add documents directly under `docs/` except for `docs/README.md`.
- Use English by default. Keep technical terms in their canonical form.
- Every document must start with one `#` title and use only the sections needed
  by its document type. Follow `docs/contributing/README.md` for document
  shapes.
- Do not add editorial feature versions such as `v1`, `v2`, or `phase 1` to
  filenames, titles, or prose. Preserve versions that are part of real API
  identifiers, formats, protocols, or shader directives.
- Use RFC-style wording with `must`, `should`, and `may` in normative contracts.
- Wrap contract names, types, functions, and parameters in backticks.
- Do not add host-local Markdown links or absolute local filesystem paths to
  documentation. Use repo-relative paths in backticks for code references, or
  repo-relative Markdown links when an actual document link is needed.
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
- Do not bind stateless, host-independent operations to runtime owners. Prefer
  module functions or static methods when behavior depends only on explicit
  inputs; use instance services when they own state, lifecycle, or injected
  dependencies. Stateless implementations may still satisfy narrow operation
  interfaces at composition boundaries.
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
2. When adding features, do not optimize solely for delivery speed. Evaluate
   architectural fit, extensibility, maintainability, and long-term tradeoffs.
3. Do not modify files, launch Playwright browser tests, or install modules
   unless the user explicitly requests that specific action.
4. Avoid large-scale refactoring unless explicitly requested.
5. Add or update regression tests for new behavior and bug fixes. Before
   creating a new test file, verify that it covers meaningful behavior, a
   regression, or an architectural contract not already covered by existing
   tests. Prefer extending an existing relevant test when it provides
   equivalent coverage. Do not add tests that only assert class or file names,
   method placement, static-versus-instance shape, or similarly simple
   implementation details.
6. When changing public APIs or behavior, update relevant `docs/` first, then
   update tests in the same change.
7. Update `AGENTS.md` only when high-priority agent guidance changes.
8. When architecture changes, audit stale descriptions in `AGENTS.md`, `docs/`,
   `README*.md`, and relevant code comments.

## Commit Message Guidelines

- Commit headers must use Conventional Commits: `type(scope)!: subject`.
- Use the smallest accurate approved type.
- Use `!` and `BREAKING CHANGE:` for breaking API, behavior, file layout,
  shader contract, or backend contract changes.
- Read `docs/contributing/README.md` before creating
  commits.
