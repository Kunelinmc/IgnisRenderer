# Migration Guidance

This document collects the upgrade actions that remain relevant to current
IgnisRenderer consumers and contributors. Completed implementation history that
does not require an action is retained in Git rather than in the documentation
tree.

## Environment Lighting Ownership

Environment spherical harmonics projection and specular prefiltering are
explicit application or tooling operations. `Renderer` no longer schedules an
environment update stage or exposes environment bake/update methods.

Replace renderer-owned environment updates with:

- `projectEnvironmentTextureToSH(...)` for spherical harmonics coefficients.
- `IBLPrefilter` or `prefilterEnvironmentIBL(...)` for specular mip chains.
- Explicit assignment of the generated data to the target probes.
- `renderer.requestRender("lighting")` after replacing probe data.

The removed surface includes `Renderer.setEnvironmentIBLUpdateOptions`,
`Renderer.getEnvironmentIBLUpdateOptions`,
`Renderer.requestEnvironmentIBLUpdate`,
`bakeEnvironmentIBLFromEnvironmentMap`, and `EnvironmentIBLBake*` types.

`IBLPrefilter` accepts its backend through `service`. Positional sources,
`IBLPrefilterBackendSource`, `IBLPrefilterSourceOptions`,
`IBLPrefilterOptions.backend`, `IBLPrefilterOptions.computeSource`, and direct
WebGPU compute sources are removed.

## WebGL Context Work

WebGL frame lifecycle operations, warmup, environment prefiltering, maintenance,
and custom render-target readback share the backend context work queue.
Applications must wait for an active renderer frame to finish before requesting
custom render-target readback.

## Post-Process Execution Declarations

Custom post-process implementations describe color, G-buffer, history,
transient, backend-shared resources, and frame requirements through one
`describeExecution()` result.

Migrate resource behavior away from graph metadata, context-binding metadata,
and descriptor-specific methods. Use the fixed declaration-checked resource
accessor during execution. There is no compatibility adapter for the removed
metadata and pass-ID execution paths.

## Shader Directive Runtime

Directive processing is owned by `ShaderRuntime`, `ShaderDirectiveStage`, and
`ShaderBackendCompileStage`. Replace `preprocessEngineShaderDirectives(...)`
and `ENGINE_DIRECTIVE_RUNTIME` with the staged runtime API.

`ShaderRule` supports `transform` and `replace`; source maps may carry
`schemaVersion` and optional column spans. Runtime caches are memory-only, so
persisted cache migration is unnecessary.

## Software Backend Lifecycle

`SoftwareBackend` participates in the same attach, initialize, frame
transaction, abort, restore, and destroy lifecycle as the GPU backends. Illegal
lifecycle calls now throw, and failed frames do not publish temporal or
post-process history.

The package-root `Rasterizer` export is removed without a public replacement.
Applications render through `Renderer` configured with `SoftwareBackend`.

## Screen-Space Global Illumination

WebGPU SSGI uses depth-based ray marching, independent temporal history, and
depth/normal-aware denoising. Remove legacy gather-mode configuration and pass
the current world-space distance and thickness settings directly. WebGL,
Software, and orthographic WebGPU cameras do not execute SSGI.

## WebGPU Deferred G-buffer

The WebGPU deferred G-buffer shader contract now uses seven compact color
attachments and two storage payloads. `gMaterialExt1` and `gMaterialExt2` are
removed. `gMaterialExt3` is `rgba16uint`, and `gMotionDepth.w` contains a packed
material word rather than a plain shading-model number.

Applications with `ShaderMaterial.deferredLighting === true` must update their
`fragment-deferred` chunks to the formats and field mapping in the
[WebGPU contract](../contracts/webgpu.md#deferred-lighting). Existing chunks
that declare four `rgba16float` storage outputs are not compatible and do not
receive a legacy adapter. TypeScript construction and entry-point APIs remain
unchanged.

Custom deferred fragments must emit color locations `0..6` in this order:
`gAlbedoAlpha`, `gNormalRoughMetal`, `gEmissiveOcclusion`, `gMotionDepth`,
`gSpecular`, `gCoatSheen`, and `gSheenReflectance`. Group `3`, binding `0` is
the write-only `rgba16float` `gMaterialExt0Out`; binding `1` is the write-only
`rgba16uint` `gMaterialExt3Out`. Custom deferred shaders always select the
extended layout. They must follow the shared material-word and extension packing
defined by `deferredGBufferCodec.wgsl`; the deferred resolve read ABI is color
bindings `0..6`, `gMaterialExt0In` at binding `7`, and `gMaterialExt3In` at
binding `8`.

## Related Documents

- [Renderer guide](../public/renderer.md)
- [Renderer contract](../contracts/renderer.md)
- [Lighting contract](../contracts/lighting.md)
- [Shader contract](../contracts/shaders.md)
- [Post-process contract](../contracts/postprocess.md)
- [WebGL contract](../contracts/webgl.md)
