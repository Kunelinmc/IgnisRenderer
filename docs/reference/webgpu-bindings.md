# WebGPU Bindings

## Scope

This document lists WebGPU resource bind groups and binding indices used by
IgnisRenderer.

The list covers:

- Core render pipeline layouts in `src/backends/webgpu/WebGPUPipelineLayouts.ts`.
- Frame, material, particle, clustered-lighting, G-buffer, shadow, present, and
  OIT bindings under `src/backends/webgpu/`, including
  `rendergraph/WebGPUFrameOrchestrator.ts` and `WebGPUShadowPass.ts`.
- WGSL bindings under `src/shaders/webgpu/`, including
  `parts/definitions.wgsl`, `parts/fragmentGBuffer.wgsl`,
  `deferredLightingShader.wgsl`, `environmentShader.wgsl`,
  `particleShader.wgsl`, `particleSimulation.wgsl`, and post-process shaders.
- WebGPU compute kernels created through `ComputeRuntime` by
  `WebGPUIBLPrefilterExecutor`,
  `SobelNormalMapper`, and `WebGPUParticleSimulator`.

This document does not list vertex attribute `shaderLocation` values.

## Background

WebGPU scene rendering uses stable bind group roles:

- `group(0)` is frame-scoped scene data for scene, particle, deferred, and
  post-process frame-aware passes.
- `group(1)` is model/material data for mesh scene passes, particle material
  data for particle render passes, or pass-specific animation data for shadow
  passes.
- `group(2)` is clustered lighting data for scene and deferred lighting passes.
- `group(3)` is G-buffer extension write or deferred lighting read data,
  depending on the pipeline layout.

Standalone compute and post-process kernels generally use `group(0)` for
pass-local resources. Some kernels use `group(1)` for the shared frame bind
group.

## Binding Tables

### Shared WGSL Math Constants

`#import <ignis/webgpu/constants>` must provide these shared `f32` constants:

| Constant | Value |
| --- | --- |
| `PI` | `3.14159265359` |
| `TWO_PI` | `6.28318530718` |
| `HALF_PI` | `1.57079632679` |
| `INV_PI` | `0.31830988618` |
| `INV_TWO_PI` | `0.15915494309` |
| `EPSILON` | `1e-6` |

Built-in shaders and custom WGSL should import this module instead of defining
duplicate equivalents. Pass-specific tolerances may remain local when their
different value is intentional.

### Core Pipeline Bind Groups

The following pipeline layouts must preserve these bind group roles:

| Pipeline layout | Group 0 | Group 1 | Group 2 | Group 3 |
| --- | --- | --- | --- | --- |
| `scenePipelineLayout` | `sceneFrameBindGroupLayout` | `modelBindGroupLayout` | `clusteredSceneBindGroupLayout` | N/A |
| `sceneGBufferPipelineLayout` | `sceneFrameBindGroupLayout` | `modelBindGroupLayout` | `clusteredSceneBindGroupLayout` | `gbufferWriteBindGroupLayout` |
| `sceneDepthPrepassPipelineLayout` | `sceneFrameBindGroupLayout` | `modelBindGroupLayout` | `clusteredSceneBindGroupLayout` | N/A |
| `deferredLightingPipelineLayout` | `sceneFrameBindGroupLayout` | `deferredUnusedBindGroupLayout` | `clusteredSceneBindGroupLayout` | `gbufferReadBindGroupLayout` |
| `environmentPipelineLayout` | `environmentFrameBindGroupLayout` | N/A | N/A | N/A |
| `particlePipelineLayout` | `sceneFrameBindGroupLayout` | `particleBindGroupLayout` | N/A | N/A |

### `sceneFrameBindGroupLayout` - `group(0)`

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `frame` | `FrameCameraUniforms` uniform buffer (288 bytes) |
| `1` | `shadowAtlas` | `texture_depth_2d` |
| `2` | `envSpecularTexture` | `texture_2d<f32>` |
| `3` | `envSpecularSampler` | Filtering sampler |
| `4` | `envSpecularFallbackTexture` | `texture_2d<f32>` |
| `5` | `envSpecularFallbackSampler` | Filtering sampler |
| `6` | `fog` | `FogUniforms` uniform buffer |
| `7` | `particleShadowVolumes` | Read-only storage buffer |
| `8` | `shadowTransmittanceAtlas` | `texture_2d<f32>` |
| `9` | `brdfLUTTexture` | `texture_2d<f32>` |
| `10` | `irradianceProbeGridCoeffs` | `texture_2d<f32>` |
| `11` | `pagedShadowPageTable` | `texture_2d<u32>` |
| `12` | `pagedShadowPhysicalDepth` | `texture_depth_2d` |
| `13` | `shadowComparisonSampler` | Comparison sampler |
| `14` | `frameLights` | `FrameLightUniforms` uniform buffer (1,680 bytes) |
| `15` | `frameShadows` | `FrameShadowUniforms` uniform buffer (5,760 bytes) |
| `16` | `frameEnvironment` | `FrameEnvironmentUniforms` uniform buffer (4,208 bytes) |

### `environmentFrameBindGroupLayout` - `group(0)`

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `frame` | `FrameCameraUniforms` uniform buffer (288 bytes) |
| `1` | `environmentTexture` | `texture_2d<f32>` |
| `2` | `environmentSampler` | Filtering sampler |
| `3` | `environmentBackground` | Environment background uniform buffer |

### `modelBindGroupLayout` - `group(1)`

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `model` | `ModelUniforms` uniform buffer |
| `1` | `baseColorTexture` | `texture_2d<f32>` |
| `2` | `baseColorSampler` | Filtering sampler |
| `3` | `metallicRoughnessTexture` | `texture_2d<f32>` |
| `4` | `metallicRoughnessSampler` | Filtering sampler |
| `5` | `normalTexture` | `texture_2d<f32>` |
| `6` | `normalSampler` | Filtering sampler |
| `7` | `emissiveTexture` | `texture_2d<f32>` |
| `8` | `emissiveSampler` | Filtering sampler |
| `9` | `occlusionTexture` | `texture_2d<f32>` |
| `10` | `occlusionSampler` | Filtering sampler |
| `11` | `specularTexture` | `texture_2d<f32>` |
| `12` | `specularSampler` | Filtering sampler |
| `13` | `specularColorTexture` | `texture_2d<f32>` |
| `14` | `specularColorSampler` | Filtering sampler |
| `15` | `clearcoatTexture` | `texture_2d<f32>` |
| `16` | `clearcoatSampler` | Filtering sampler |
| `17` | `clearcoatRoughnessTexture` | `texture_2d<f32>` |
| `18` | `clearcoatRoughnessSampler` | Filtering sampler |
| `19` | `clearcoatNormalTexture` | `texture_2d<f32>` |
| `20` | `clearcoatNormalSampler` | Filtering sampler |
| `21` | `sheenColorTexture` | `texture_2d<f32>` |
| `22` | `sheenColorSampler` | Filtering sampler |
| `23` | `sheenRoughnessTexture` | `texture_2d<f32>` |
| `24` | `sheenRoughnessSampler` | Filtering sampler |
| `25` | `transmissionTexture` | `texture_2d<f32>` |
| `26` | `transmissionSampler` | Filtering sampler |
| `27` | `thicknessTexture` | `texture_2d<f32>` |
| `29` | `iridescenceTexture` | `texture_2d<f32>` |
| `30` | `animationParams` | `AnimationParams` uniform buffer |
| `31` | `iridescenceThicknessTexture` | `texture_2d<f32>` |
| `32` | `jointMatrices` | Read-only storage buffer |
| `33` | `morphWeights` | Read-only storage buffer |
| `34` | `morphPositionDeltas` | Read-only storage buffer |
| `35` | `morphNormalDeltas` | Read-only storage buffer |
| `36` | `shaderUniforms` | Optional material shader uniform buffer |
| `37` | `anisotropyTexture` | `texture_2d<f32>` |

Material texture slots are defined by `WEBGPU_TEXTURE_SLOT`. Texture bindings
are `1 + slot * 2`. Dedicated sampler bindings are `2 + slot * 2` only for
slots `0..12`. `thicknessTexture`, `iridescenceTexture`,
`iridescenceThicknessTexture`, and `anisotropyTexture` do not have dedicated
sampler bindings; shader code samples them with `transmissionSampler`.
`anisotropyTexture` uses binding `37` and is not a
`WEBGPU_TEXTURE_SLOT` entry.

### `clusteredSceneBindGroupLayout` - `group(2)`

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `clusterGrid` | `ClusterGridParams` uniform buffer |
| `1` | `clusterLights` | Read-only storage buffer |
| `2` | `clusterHeaders` | Read-only storage buffer |
| `3` | `clusterIndices` | Read-only storage buffer |

### Clustered Lighting Compute

`clusteredLightingCull.wgsl` uses `csClear`, `csScatter`, and `csFinalize`
compute entry points with a shared compute-specific layout. `ClusterGridParams`
stores `maxLightsPerCluster` in its final `u32` slot so compute and fragment
passes agree on the fixed index span for each cluster.

| Group | Binding | Shader name | Resource contract |
| --- | --- | --- | --- |
| `0` | `0` | `clusterParams` | `ClusterGridParams` uniform buffer |
| `0` | `1` | `clusterLights` | Read-only storage buffer |
| `0` | `2` | `clusterHeaders` | Read-write storage buffer |
| `0` | `3` | `clusterIndices` | Read-write storage buffer |
| `1` | `0` | `frame` | `FrameCameraUniforms` uniform buffer through `sceneFrameBindGroupLayout` |

### G-buffer Bindings

`gbufferWriteBindGroupLayout` is used as `group(3)` by
`sceneGBufferPipelineLayout`:

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `gMaterialExt0Out` | Write-only `rgba16float` storage texture |
| `1` | `gMaterialExt3Out` | Write-only `rgba16uint` storage texture |

`gbufferReadBindGroupLayout` is used as `group(3)` by
`deferredLightingPipelineLayout`:

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `gAlbedoAlphaIn` | `texture_2d<f32>` |
| `1` | `gNormalRoughMetalIn` | `texture_2d<f32>` |
| `2` | `gEmissiveOcclusionIn` | `texture_2d<f32>` |
| `3` | `gMotionDepthIn` | `texture_2d<f32>` |
| `4` | `gSpecularIn` | `texture_2d<f32>` |
| `5` | `gCoatSheenIn` | `texture_2d<f32>` |
| `6` | `gSheenReflectanceIn` | `texture_2d<f32>` |
| `7` | `gMaterialExt0In` | `texture_2d<f32>` |
| `8` | `gMaterialExt3In` | `texture_2d<u32>` |

### Particle Render Bindings

Particle render pipelines use `sceneFrameBindGroupLayout` as `group(0)` and
`particleBindGroupLayout` as `group(1)`.

| Group | Binding | Shader name | Resource contract |
| --- | --- | --- | --- |
| `0` | `0` | `frame` | `FrameCameraUniforms` uniform buffer |
| `0` | `1` | `shadowAtlas` | `texture_depth_2d` |
| `0` | `2` | `envSpecularTexture` | `texture_2d<f32>` |
| `0` | `3` | `envSpecularSampler` | Filtering sampler |
| `0` | `6` | `fog` | `FogUniforms` uniform buffer |
| `0` | `7` | `particleShadowVolumes` | Read-only storage buffer |
| `0` | `13` | `shadowComparisonSampler` | Comparison sampler |
| `0` | `14` | `frameLights` | `FrameLightUniforms` uniform buffer |
| `0` | `15` | `frameShadows` | `FrameShadowUniforms` uniform buffer |
| `0` | `16` | `frameEnvironment` | `FrameEnvironmentUniforms` uniform buffer |
| `1` | `0` | `particleTexture` | `texture_2d<f32>` |
| `1` | `1` | `particleSampler` | Filtering sampler |
| `1` | `2` | `particleUVTransform` | Particle UV transform uniform buffer |

### Particle Simulation Compute

`particleSimulation.wgsl` declares the full simulation layout:

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `particles` | Read-write storage buffer |
| `1` | `instances` | Read-write storage buffer |
| `2` | `drawArgs` / `indirect` | Read-write storage buffer |
| `3` | `params` | `ParticleSimParams` uniform buffer |

`WebGPUParticleSimulator` creates narrower kernels with the same binding
indices:

| Kernel | Bound resources |
| --- | --- |
| `WebGPUParticleReset` | `2 indirect` |
| `WebGPUParticleSpawn` | `0 particles`, `3 params` |
| `WebGPUParticleSimulate` | `0 particles`, `1 instances`, `2 indirect`, `3 params` |

### Shadow Pass Bindings

`WebGPUShadowPass` uses a dedicated shadow pipeline layout:

| Group | Binding | Shader name | Resource contract |
| --- | --- | --- | --- |
| `0` | `0` | `shadowMvps` | Read-only storage buffer of shadow MVP matrices |
| `0` | `1` | `shadowInstances` | Read-only storage buffer of shadow instance data |
| `0` | `2` | `shadowTransmittance` | Read-only storage buffer |
| `1` | `0` | `animationParams` | `AnimationParams` uniform buffer |
| `1` | `1` | `jointMatrices` | Read-only storage buffer |
| `1` | `2` | `morphWeights` | Read-only storage buffer |
| `1` | `3` | `morphPositionDeltas` | Read-only storage buffer |

### Present and OIT Resolve Bindings

`WebGPUFrameOrchestrator` uses these pass-local `group(0)` bindings:

| Pass | Binding | Shader name | Resource contract |
| --- | --- | --- | --- |
| Present | `0` | `srcTexture` | `texture_2d<f32>` |
| Present | `1` | `srcSampler` | Sampler |
| Present | `2` | `presentParams` | `PresentParams` uniform buffer |
| OIT resolve | `0` | `sceneColorTexture` | `texture_2d<f32>` |
| OIT resolve | `1` | `oitAccumTexture` | `texture_2d<f32>` |
| OIT resolve | `2` | `oitRevealTexture` | `texture_2d<f32>` |
| OIT resolve | `3` | `linearSampler` | Sampler |

### IBL Prefilter Compute

`iblPrefilter.wgsl` and `WebGPUIBLPrefilterExecutor` use:

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `envSampler` | Sampler |
| `1` | `envTexture` | `texture_2d<f32>` |
| `2` | `outputTexture` | Write-only `rgba16float` storage texture |
| `3` | `params` | `PrefilterParams` uniform buffer |

### Sobel Normal Compute

`sobelNormal.wgsl` and `SobelNormalMapper` use:

| Binding | Shader name | Resource contract |
| --- | --- | --- |
| `0` | `src` / `srcTexture` | `texture_2d<f32>` |
| `1` | `dst` / `dstTexture` | Write-only `rgba8unorm` storage texture |
| `2` | `params` | `Params` uniform buffer |

### Post-process Compute Bindings

All entries in this section are pass-local `group(0)` unless stated otherwise.

| Shader / pass | Bindings |
| --- | --- |
| `toneMapping.wgsl` | `0 srcTex` texture, `1 outTex` write storage texture |
| `colorFilter.wgsl` | `0 srcTex` texture, `1 linearSampler` sampler, `2 params` uniform, `3 outTex` write storage texture |
| `fxaa.wgsl` | `0 srcTex` texture, `1 linearSampler` sampler, `2 params` uniform, `3 outTex` write storage texture |
| `fog.wgsl` | `0 srcTex` texture, `1 gMotionDepth` texture, `2 linearSampler` sampler, `3 params` uniform, `4 outTex` write storage texture |
| `motionBlur.wgsl` | `0 srcTex` texture, `1 gMotionDepth` texture, `2 linearSampler` sampler, `3 params` uniform, `4 outTex` write storage texture |
| `dof.wgsl` | `0 srcTex` texture, `1 gMotionDepth` texture, `2 linearSampler` sampler, `3 params` uniform, `4 outTex` write storage texture |
| `bloomDownsample.wgsl` | `0 srcTex` texture, `1 linearSampler` sampler, `2 params` uniform, `3 dstTex` write storage texture |
| `bloomBlurH.wgsl` | `0 srcTex` texture, `1 linearSampler` sampler, `2 params` uniform, `3 dstTex` write storage texture |
| `bloomBlurV.wgsl` | `0 srcTex` texture, `1 linearSampler` sampler, `2 params` uniform, `3 dstTex` write storage texture |
| `bloomUpsample.wgsl` | `0 srcTex` texture, `1 blendTex` texture, `2 linearSampler` sampler, `3 params` uniform, `4 dstTex` write storage texture |
| `bloomComposite.wgsl` | `0 sceneTex` texture, `1 bloomTex` texture, `2 linearSampler` sampler, `3 params` uniform, `4 dstTex` write storage texture |
| `ssao.wgsl` | `0 texA` texture, `1 texB` texture, `2 linearSampler` sampler, `3 params` uniform, `4 outTex` write storage texture |
| `ssgi.wgsl` trace/temporal | `0 sceneColor` texture, `1 gNormalRoughMetal` texture, `2 gMotionDepth` texture, `3 hiZ` texture, `4 ssgiHistory` texture, `5 motionHistory` texture, `6 linearSampler` sampler, `7 traceParams` uniform, `8 outSSGIHistory` write storage texture |
| `ssgi.wgsl` trace frame group | `group(1) binding(0) frame` uniform. Runtime supplies the shared scene frame bind group. |
| `ssgi.wgsl` denoise | `0 sourceSSGI` texture, `1 gNormalRoughMetal` texture, `2 gMotionDepth` texture, `3 linearSampler` sampler, `4 denoiseParams` uniform, `5 outDenoised` write storage texture |
| `ssgi.wgsl` compose | `0 composeScene` texture, `1 composeSSGI` texture, `2 composeAlbedo` texture, `3 composeNormalRoughMetal` texture, `4 composeMotionDepth` texture, `5 composeSampler` sampler, `6 composeParams` uniform, `7 composeOut` write storage texture |
| `taa.wgsl` | `0 currentColor` texture, `1 historyColor` texture, `2 motionDepth` texture, `3 motionHistory` texture, `4 linearSampler` sampler, `5 params` uniform, `6 outColor` write storage texture, `7 outHistory` write storage texture |
| `hiz.wgsl` depth seed | `0 depthTex` texture, `1 outTex` write storage texture |
| `hiz.wgsl` mip downsample | `0 srcTex` texture, `1 dstTex` write storage texture |
| `ssr.wgsl` trace | `0 sceneColor` texture, `1 gNormalRoughMetal` texture, `2 gMotionDepth` texture, `3 hiZ` texture, `4 ssrHistory` texture, `5 motionHistory` texture, `6 linearSampler` sampler, `7 traceParams` uniform, `8 outSSR` write storage texture |
| `ssr.wgsl` trace frame group | `group(1) binding(0) frame` uniform. Runtime supplies the shared scene frame bind group. |
| `ssr.wgsl` compose | `0 composeScene` texture, `1 composeSSR` texture, `2 composeMotionDepth` texture, `3 composeSampler` sampler, `4 composeParams` uniform, `5 composeOut` write storage texture |
| `screenSpaceRefractions.wgsl` trace | `0 backgroundColor` texture, `1 transmissionSurface0` texture, `2 transmissionSurface1` texture, `3 transmissionSurface2` texture, `4 opaqueMotionDepth` texture, `5 hiZ` texture, `6 linearSampler` sampler, `7 traceParams` uniform, `8 outRefraction` write storage texture, `9 opaqueNormal` texture |
| `screenSpaceRefractions.wgsl` trace frame group | `group(1) binding(0) frame` uniform. Runtime supplies the shared scene frame bind group. |
| `screenSpaceRefractions.wgsl` compose | `0 composeScene` texture, `1 composeRefraction` texture, `2 composeTransmissionLighting` texture, `3 composeSampler` sampler, `4 composeParams` uniform, `5 composeOut` write storage texture |
| `volumetric.wgsl` | `0 sceneColor` texture, `1 gMotionDepth` texture, `2 hiZ` texture, `3 volumetricHistory` texture, `4 motionHistory` texture, `5 linearSampler` sampler, `6 params` uniform, `7 outColor` write storage texture, `8 outHistory` write storage texture, `9 volumetricReservoirHistory` texture, `10 outReservoirHistory` write storage texture, `11 volumetricLightBuffer` read-only storage buffer |
| `volumetric.wgsl` frame group | `group(1) binding(0) frame` uniform. Runtime supplies the shared scene frame bind group. |

### ComputeRuntime Dynamic Kernel Contract

`ComputeRuntime` kernels must define `bindings` in
`ComputeKernelDescriptor`. The descriptor controls `group(0)` binding order by
`key`, `binding`, and `type`.

The supported `type` values are:

| Type | Resource contract |
| --- | --- |
| `buffer` | `IRenderBuffer` |
| `texture` | `IRenderTexture` |
| `sampler` | `ISampler` |

`extraBindGroups` may bind additional groups with index greater than `0`.
`extraBindGroups` must not target group `0`, because group `0` is managed by
the kernel schema.

## Maintenance

Use this command to verify WGSL binding declarations:

```bash
rg -n "@group\\([0-9]+\\) @binding\\([0-9]+\\)" src/backends/webgpu src/shaders/webgpu -g "*.ts" -g "*.wgsl"
```

Use this command to verify TypeScript bind group layout entries:

```bash
rg -n "createBindGroupLayout|createBindingGroup|binding:" src/backends/webgpu src/addons/SobelNormalMapper.ts src/simulation/particles/WebGPUParticleSimulator.ts -g "*.ts"
```

When adding a new binding, update the WGSL declaration, the TypeScript layout or
kernel schema, the resource population site, and this document in the same
change.

## Diagnostics

- A shader validation error for a missing binding is triggered when WGSL
  declares a binding that is absent from the pipeline layout.
- A bind group creation error is triggered when TypeScript provides a resource
  whose binding index is absent from the layout.
- A resource type error is triggered when a buffer, texture, sampler, or storage
  texture does not match the WGSL declaration and layout entry.
- A sampler limit error can be triggered if additional material samplers are
  added without accounting for `WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT`.
- `ComputeRuntime` throws when a kernel schema has duplicate keys, duplicate
  binding indices, unsupported binding types, missing required resources, or
  `extraBindGroups` targeting group `0`.

## Compatibility

Changing any binding index, group index, or resource type is a breaking shader
contract change. A compatible change may add a binding only when every affected
pipeline layout, WGSL declaration, bind group population site, warmup path, and
fallback resource path are updated together.

The former monolithic `FrameUniforms` binding is split across
`FrameCameraUniforms` at binding `0`, `FrameLightUniforms` at binding `14`,
`FrameShadowUniforms` at binding `15`, and `FrameEnvironmentUniforms` at
binding `16`. Custom WGSL that reads lights, shadows, or probe data through
`frame` must migrate those reads to `frameLights`, `frameShadows`, or
`frameEnvironment`; camera and global setting fields remain on `frame`.

## Related Documents

- [WebGPU architecture](../architecture/webgpu.md)
- [WebGPU contract](../contracts/webgpu.md)
- [Shader contract](../contracts/shaders.md)
- [Compute runtime guide](../public/compute-runtime.md)
