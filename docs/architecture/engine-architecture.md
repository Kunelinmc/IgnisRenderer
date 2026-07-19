# Engine Architecture

## Scope

This document defines the architecture context for contributors changing core
renderer contracts, backend ownership, ECS synchronization, simulation stages,
foundation utilities, or worker infrastructure.

## Background

This document is the canonical architecture overview for contributor-facing
engineering decisions. Automation-specific entrypoints, such as `AGENTS.md`,
may reference this file, but the contracts here apply to all contributions.

IgnisRenderer is a TypeScript rendering engine with three primary backend
families:

- `SoftwareBackend`: a CPU scanline rasterizer with backend-owned passes for
  rasterization, light evaluation, and post-processing.
- `WebGPUBackend`: a hardware-accelerated backend with delegated registries for
  resources, bindings, and frame execution.
- `WebGLBackend`: a modern compatibility backend using the current V1/V2 WebGL
  architecture documented by backend-specific contracts.

The scene graph is backed by an Entity Component System. `Node` is the
application-facing facade, and ECS components store simulation-ready state.
Animation, physics, and particle simulation integrate into the renderer pipeline
through dedicated simulation stages and backend-owned runtimes.

## API/Contract

### Backend Instance Contract

- `IRenderBackend.profile.id` must identify the backend implementation.
- `IRenderBackend.attach(context)` must bind a backend instance to at most one
  `Renderer`.
- Backend instances must expose profile data, capabilities, extensions, device
  lifecycle, and frame execution on the backend instance.
- A second renderer must receive a new backend instance.
- Backend device lifecycle and resource ownership must not leak into
  backend-agnostic public contracts.
- WebGPU device initialization must create a narrow device-scoped frame host.
  Backend-internal frame runtimes must depend on that host rather than the full
  `WebGPUBackend` lifecycle facade.
- WebGL device initialization must create a context-scoped frame service owner.
  WebGL frame graph nodes and post-process adapters must depend on narrow
  service contracts rather than the full `WebGLBackend` or concrete frame
  executor facade.
- Backend-native resources must have one lifecycle owner. Aggregate frame
  facades may coordinate owners but must not duplicate native handle lifetime.
- Device loss must destroy backend-owned post-process state before invalidating
  the frame host and destroying frame/shared GPU resources.

### ECS and Scene Graph Contract

- `ECSWorld` must manage entities and components such as `LocalTransform`,
  `WorldTransform`, `NodeRef`, and `Visibility`.
- Systems should use `world.query(["CompA", "CompB"])` for efficient component
  filtering.
- `Node` must remain a high-level scene graph object synchronized
  bi-directionally with ECS entities.
- `Scene.syncNodeToECS` must transfer manual `Node` changes into ECS components
  during sync-in.
- `Scene.syncECSToNode` must propagate ECS simulation output back to `Node`
  facades during sync-out.
- Definition layers, such as interfaces and types, must stay separate from logic
  layers, such as systems and simulation stages.

### Simulation Contract

- `AnimationRuntime` owns animation mixers, blend trees, and skeletal update
  integration. `FrameCoordinator` owns animation stage scheduling within the
  renderer frame pipeline.
- `PhysicsSystem` owns collision detection and rigid body dynamics through
  adapter contracts.
- `DefaultParticleSimulator` and `WebGPUParticleSimulator` own high-density
  particle state updates through backend-owned `particle-sim` passes.
- Simulation time steps must use seconds and should use names ending in
  `deltaTimeSeconds`.
- Hot simulation paths should use pre-allocated math objects instead of
  avoidable `new` allocations.

### Foundation and Utility Contract

- `src/foundation/Color.ts` owns HSL/RGB parsing and linear-space color
  utilities.
- `src/foundation/Error.ts` must define custom error subclasses and their
  initialization/data types.
- Subsystem files must not define local custom error subclasses. They must
  import centralized custom errors from `src/foundation/Error.ts`.
- Ordinary `throw new Error(...)` usage may remain local when no custom error
  class is needed.
- `src/foundation/IdGenerator.ts` owns deterministic ID generation.
- `src/foundation/Platform.ts` owns environment detection and browser-specific
  capability checks.

### Worker Infrastructure Contract

- `src/workers/WorkerScheduler` owns the Web Worker pool for parallel tasks such
  as environment IBL prefiltering and worker-backed physics adapters.
- `src/workers/transports.ts` owns efficient zero-copy transport behavior using
  `SharedArrayBuffer` where available.
- Worker-facing payload contracts must avoid backend-specific public API leaks.

## Usage

Review this document before changing ECS synchronization, simulation stages,
backend lifecycle ownership, foundation utilities, or worker transport
contracts.

Example validation commands:

```bash
bunx tsc --noEmit
bun run test
bun tests/static/physics/test_physics_adapter_contract.mjs
```

## Errors & Diagnostics

- Backend reuse errors must indicate that backend instances are one-shot
  renderer runtimes.
- ECS synchronization regressions should be covered by targeted tests that
  verify sync-in and sync-out behavior.
- New custom error classes must be discoverable in `src/foundation/Error.ts`.
- Worker transport failures should report whether `SharedArrayBuffer` support or
  transfer ownership constraints triggered the failure.

## Compatibility / Breaking Changes

Changes to backend instance lifecycle, ECS component names, `Node`
synchronization semantics, simulation stage ownership, or worker transport
payloads may be breaking changes. Such changes must update corresponding
contract documents, tests, and migration notes in the same PR.

`SoftwareBackend` uses a single scanline rasterization path. The former tile
raster mode and its worker-binning payloads are removed rather than retained as
a backend execution alternative.
