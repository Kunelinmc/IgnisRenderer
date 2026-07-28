# IgnisRenderer

IgnisRenderer is a high-performance 3D rendering engine written in TypeScript.
It provides a shared renderer API across software, WebGL, and WebGPU backends
while keeping backend resource ownership and execution details isolated.

## Features

- Software, WebGL, and WebGPU rendering through a shared API.
- Physically based materials, lighting, shadows, and post-processing.
- Scene loading, cameras, meshes, materials, and animation.
- Optional physics and particle simulation.
- Custom render targets and render passes for advanced workflows.

## Rendering Backends

IgnisRenderer offers three backends for different environments. Applications
use the same high-level renderer API and select the backend that best fits
their target platform.

- **Software** renders on the CPU. Its main goal is to work without a modern
  graphics API and provide a predictable implementation for compatibility,
  testing, and CPU-based rendering.
- **WebGL** targets broad browser and device support. It uses established GPU
  APIs to provide hardware-accelerated scene rendering, lighting, shadows, and
  post-processing.
- **WebGPU** targets modern browsers and GPUs. It is designed for higher-end
  rendering features, more flexible GPU workloads, and better performance on
  current hardware.

Feature availability may vary by backend and device capability.

## Architecture

`Renderer` provides the main application interface. Scene objects are organized
through a familiar node hierarchy, while the engine uses an ECS internally for
efficient updates. Rendering, animation, physics, and particles are coordinated
through a shared frame pipeline before work is passed to the selected backend.

## Project Structure

```text
IgnisRenderer/
|-- src/
|   |-- core/          Scene objects and shared resources
|   |-- rendering/     Public renderer interface
|   |-- backends/      Software, WebGL, and WebGPU implementations
|   |-- pipeline/      Frame preparation and rendering flow
|   |-- rendergraph/   Shared render graph building and compilation
|   |-- postprocess/   Screen-space visual effects
|   |-- shaders/       Shader programs for each backend
|   |-- ecs/           Internal scene data storage
|   |-- simulation/    Shared simulation flow
|   |-- animation/     Animation systems
|   |-- physics/       Physics integration
|   |-- particles/     Particle effects
|   |-- loaders/       Asset and glTF loading
|   |-- materials/     Surface appearance definitions
|   |-- meshes/        Geometry data
|   |-- cameras/       Scene viewpoints
|   |-- lights/        Scene lighting
|   |-- maths/         Math utilities
|   |-- workers/       Background task support
|   |-- foundation/    Shared engine utilities
|   `-- addons/        Optional and experimental features
|-- tests/
|   |-- static/        Headless and contract tests
|   |-- browser/       Browser integration tests
|   `-- benchmarks/    Performance benchmarks
|-- docs/              Public API and contributor documentation
|-- scripts/           Build and maintenance scripts
`-- assets/            Images and sample assets
```

See the [documentation index](docs/README.md) for public API guidance,
architecture, internal contracts, and contribution workflows.

## License

IgnisRenderer is distributed under the MIT License. See `LICENSE` for the full
license text.
