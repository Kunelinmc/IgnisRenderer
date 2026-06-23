# Occlusion Culling Contract
## Scope
This document defines the v1 contract for renderer occlusion culling feature
negotiation, prepared-scene filtering, WebGPU runtime execution, diagnostics,
and backend compatibility.

## Background
Occlusion culling reduces main-camera opaque draw cost by hiding draw packets
that were proven occluded by depth produced in a previous frame. In v1, the
feature is WebGPU-first and uses a previous-frame Hi-Z visibility snapshot:
the prepared-scene build stage must use the latest completed CPU snapshot, and
the WebGPU backend must asynchronously produce results for a later frame.

The prepared-scene build stage must not wait for GPU work. Missing, stale, or
untrusted visibility information must keep objects visible.

## API/Contract
- `RendererFeatureRequest.enableOcclusionCulling` may request occlusion
  culling and must default to `false`.
- `RendererFeatureRequest.occlusionCullingOptions` may override
  `OcclusionCullingOptions`.
- `DEFAULT_OCCLUSION_CULLING_OPTIONS` must define:
  - `minCandidateScreenAreaPx = 64`.
  - `minOccluderScreenAreaPx = 256`.
  - `hysteresisFrames = 2`.
  - `maxReadbackLatencyFrames = 3`.
  - `debug = false`.
- `BackendCapabilities.occlusionCulling` must exist on all backends.
- `WebGPUBackend.profile.capabilities.occlusionCulling` must be `true` unless
  `WebGPUBackendOptions.enableOcclusionCulling === false`.
- `WebGLBackend.profile.capabilities.occlusionCulling` must be `false` in v1.
- `SoftwareBackend.profile.capabilities.occlusionCulling` must be `false` in v1.
- Backends that support occlusion culling must expose an
  `OcclusionCullingBackendAdapter` through the `renderer.occlusion-culling`
  backend extension.
- `Renderer` must resolve occlusion culling integration with
  `resolveOcclusionCullingBackendExtension(backend)?.api`.
- `IRenderBackend` must not expose `occlusionCullingAdapter`.
- `resolveFeatureState(...)` must disable `enableOcclusionCulling` when the
  backend capability is `false` and must emit a feature warning.
- `PreparedScene.occlusion` may expose prepared-scene occlusion metadata,
  including candidates, culled packet ids, statistics, and the visibility
  source frame.
- `OcclusionVisibilityProvider` must provide synchronous snapshot queries only.
  It must not perform asynchronous GPU waits during prepared-scene building.
- `PreparedSceneBuilder` must only filter main-camera `opaquePackets`.
- `PreparedSceneBuilder` must not filter `transparentPackets`,
  `shadowCasterPackets`, `shadowTransmitterPackets`, reflection captures, or
  probe captures in v1.
- Decal packet generation must run after opaque packet filtering, so hidden
  opaque receivers do not create decal work.
- WebGPU occlusion culling must use backend-internal frame graph nodes and must
  not add renderer-level global pipeline stages.
- The WebGPU internal frame graph may create an `occlusion-test` node only when
  occlusion culling is enabled and the prepared scene has eligible candidates.
- The `occlusion-test` node must execute after opaque depth or deferred depth is
  available.
- WebGPU occlusion culling must rely on a sampled depth-like source. In v1, the
  runtime must prefer the frame-target path that provides `gMotionDepth`.
- Missing `gMotionDepth`, missing candidates, device restore, resize, camera
  reset, stale results, missing results, and packet signature changes must make
  affected candidates visible.
- A candidate may be hidden only after `hysteresisFrames` consecutive occluded
  GPU results.
- A single visible GPU result must immediately make the candidate visible.
- v1 eligibility must exclude blend, transmission, wireframe,
  `depthWrite = false`, custom shader, alpha-mask, and unsupported topology
  packets.

## Usage
```ts
import { Renderer } from "../src/renderers/Renderer";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend";

const backend = new WebGPUBackend();
const renderer = new Renderer(backend, canvas, camera);
await renderer.initialize();

renderer.features.enableOcclusionCulling = true;
renderer.features.occlusionCullingOptions = {
	hysteresisFrames: 2,
	maxReadbackLatencyFrames: 3,
};
await renderer.renderFrame(performance.now());
```

```ts
const disabledBackend = new WebGPUBackend({
	enableOcclusionCulling: false,
});
const disabledRenderer = new Renderer(disabledBackend, canvas, camera);

console.assert(
	disabledRenderer.backendProfile.capabilities.occlusionCulling === false
);
```

```bash
bunx tsc --noEmit
bun tests/static/pipeline/test_render_list_builder.mjs
bun tests/static/pipeline/test_prepared_scene_cache.mjs
bun tests/static/renderer/test_backend_extensions.mjs
bun tests/static/webgpu/test_webgpu_frame_graph_planner.mjs
bun tests/static/webgpu/test_webgpu_occlusion_culling_runtime.mjs
```

## Errors & Diagnostics
- `{backend}-feature-occlusion-culling` must be emitted once when
  `enableOcclusionCulling` is requested on a backend whose
  `BackendCapabilities.occlusionCulling` is `false`.
- `webgpu-occlusion-hiz-failed` may be emitted when the WebGPU runtime cannot
  build the Hi-Z texture for the current frame.
- `webgpu-occlusion-encode-failed` may be emitted when the WebGPU runtime cannot
  record the visibility compute pass.
- `webgpu-occlusion-readback-failed` may be emitted when an asynchronous
  readback cannot be mapped or consumed.

All runtime failures must fall back to visible candidates for the affected
frame or snapshot. Diagnostics should use warn-once behavior where repeated
frames would otherwise produce duplicate warnings.

## Compatibility / Breaking Changes
Occlusion culling remains opt-in and defaults to disabled for renderer feature
users. Backend integration APIs are breaking:
`backend.occlusionCullingAdapter` is removed and must be replaced with
`renderer.getBackendExtension(OCCLUSION_CULLING_EXTENSION)`. Unsupported backends
must keep existing rendering behavior by disabling the resolved feature.

WebGPU may promote an active occlusion-culling frame away from the single-target
path so `gMotionDepth` can be sampled by the occlusion runtime. This is a
backend-internal target selection detail and must not require public
post-process graph APIs.
