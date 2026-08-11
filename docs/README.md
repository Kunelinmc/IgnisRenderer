# IgnisRenderer Documentation

IgnisRenderer documentation is organized by audience and durable ownership.
Start with a public workflow when using the package, or with architecture and
contracts when changing the engine.

## Package Consumers

- [Renderer](public/renderer.md): renderer creation, frames, post-processing,
  backend events, sizing, and cleanup.
- [Interaction](public/interaction.md): picking, hover, selection, drag
  selection, and transform controls.
- [Compute Runtime](public/compute-runtime.md): application-defined WebGPU
  compute resources, kernels, dispatch, and readback.

## Architecture

- [Engine architecture](architecture/engine.md): engine layers, ECS, simulation,
  backend ownership, foundation services, and workers.
- [Rendering architecture](architecture/rendering.md): portable frame flow,
  data conventions, shaders, lighting, materials, and post-processing.
- [Render Graph architecture](architecture/render-graph.md): whole-frame logical
  graph composition, analysis, compilation, and attempt tracking.
- [WebGPU architecture](architecture/webgpu.md): WebGPU frame planning,
  execution, resources, submission, and failure boundaries.

## Contracts

- [Renderer](contracts/renderer.md)
- [Compute](contracts/compute.md)
- [Geometry](contracts/geometry.md)
- [Physics](contracts/physics.md)
- [Loaders](contracts/loaders.md)
- [Rendering features](contracts/rendering.md)
- [Lighting](contracts/lighting.md)
- [Shadows](contracts/shadows.md)
- [Materials](contracts/materials.md)
- [Particles](contracts/particles.md)
- [Shaders](contracts/shaders.md)
- [Post-processing](contracts/postprocess.md)
- [Software backend](contracts/software.md)
- [WebGPU backend](contracts/webgpu.md)
- [WebGL backend](contracts/webgl.md)

## Maintainer Resources

- [WebGPU bindings reference](reference/webgpu-bindings.md)
- [Migration guidance](migrations/README.md)
- [WebGL physical lighting migration](migrations/webgl-physical-lighting.md)
- [Contributing documentation](contributing/README.md)

## Documentation Policy

The documentation tree stays shallow: maintained Markdown files live at
`docs/<category>/<document>.md`. Small features are sections of their owning
subsystem document rather than standalone files. Architecture explains design;
contracts contain normative behavior. See the
[contributing documentation](contributing/README.md) for authoring and commit
rules.
