# Early-Z Prepass Contract

## Scope

This document defines the Early-Z depth prepass contract for scene rendering, including API options, pass scopes, material eligibility (such as alpha mask discard and depth-write behaviors), shader/pipeline requirements, and backend-specific details across the `SoftwareBackend`, `WebGLBackend`, and `WebGPUBackend`.

## Background

Fragment shading costs increase with overdraw. A depth-only prepass reduces redundant fragment shading by writing eligible opaque geometry to the scene depth buffer first, then running the main color (or G-buffer geometry) pass against that depth buffer to reject covered fragments before shading.

This optimization must remain a backend-internal sequence and must not introduce a global renderer-level frame stage.

## API/Contract

### Configuration Contract

The option `enableEarlyZPrepass?: boolean` is accepted by `SoftwareBackendOptions`, `WebGLBackendOptions`, and `WebGPUBackendOptions`.

- **Input contract**: Accepts `true`, `false`, or `undefined`.
- **Default behavior**: `undefined` must be treated as `true`.
- **Behavior contract**:
	- When `true`, the main opaque path must execute a depth-only prepass before color/G-buffer shading.
	- When `false`, the main opaque path must run without the prepass and must keep the legacy single-pass depth test and write behavior.

### WebGL-Specific APIs

- `WebGLBackend.isEarlyZPrepassEnabled()` must return the resolved backend option.
- The WebGL backend capability `occlusionCulling` must remain `false`.
- `ShaderMaterial.chunks` may contain a WebGL depth prepass chunk with `backend: "webgl"`, `language: "glsl"`, and `stage: "fragment-depth"`.
- `ShaderMaterial.resolveWebGLDepthPrepassProgram(mode, options)` must return a custom WebGL vertex and depth fragment program when both vertex and depth-fragment chunks are available. It must return `null` when the material has not opted in.

### Pass Scope & Eligibility Contract

- The Early-Z prepass must apply only to `main-opaque`.
- `main-transparent`, OIT passes, transmission passes, particles, and environment rendering must not use the prepass.
- **WebGPU Deferred Path**: When deferred lighting is active, the Early-Z prepass must run before the G-buffer geometry pass and must not run before the fullscreen deferred lighting resolve pass.
- **Depth-Write Disable**: Materials with `depthWrite === false` must not participate in the prepass. Their color draws must not use the read-only depth state since they were not prepassed.

### Material Mask / Alpha Discard Contract

- Opaque mask materials (`material.alphaMode === MASK`) must run alpha-test discard in the depth prepass and must not write color targets.
- **Software Backend**: Triangles with `material.alphaMode === MASK` must be skipped in the prepass and must be evaluated in the regular color pass.
- **WebGL Backend**: Built-in material depth prepass fragments must only apply alpha mask discard. The discard test must use `uBaseColor.a * texture(uBaseMap, uv).a < uAlpha.x` when `uHasBaseMap == 1`, and `uBaseColor.a < uAlpha.x` otherwise. A WebGL `ShaderMaterial` using `AlphaMode.Mask` without a WebGL `fragment-depth` chunk should warn once and skip the Early-Z prepass for that material.
- **WebGPU Backend**: A `ShaderMaterial` with `alphaMode === MASK` must provide explicit depth prepass fragment shader configuration via `depthFragmentCode` and `depthFragmentEntryPoint`. If this configuration is missing, the engine must skip the Early-Z prepass for that material and log a warning once.

### Pipeline and Render State Contract

- **WebGPU Backend**:
	- Opaque non-mask prepass must use depth-only pipeline state with `depthWriteEnabled = true` and `depthCompare = less`.
	- Opaque color draws that were prepassed must use read-only depth state with `depthWriteEnabled = false` and `depthCompare = less-equal` (`early-z-color`).
	- Opaque color draws that were not prepassed must keep legacy depth state (`depthWriteEnabled = material.depthWrite !== false` and `depthCompare = less`).
	- G-buffer draws that were prepassed must use the same read-only `early-z-color` depth state as legacy MRT color draws.
- **WebGL Backend**:
	- The depth prepass must use depth writes, `LESS` depth test, disabled blending, and disabled color writes.
	- The following color pass must use `LEQUAL` depth test and read-only depth only for packet IDs submitted by the depth prepass.
- **Software Backend**:
	- During color shading, when the early-depth buffer is present, the early gate must use `zCamValue <= earlyDepthBuffer[pixel]`.
	- Final visibility write to `attachments.depthBuffer` must keep strict `shadedDepth < depthBuffer[pixel]`.

### Buffer and Clears Contract

- **Software Backend**:
	- The implementation must maintain an internal reusable early-depth buffer.
	- Full-frame rendering must clear the full early-depth buffer region to `Infinity` before prepass writes.
	- Incremental rendering must clear dirty rect regions to `Infinity` and may preserve non-dirty regions.
- **WebGPU Backend**:
	- Incremental dirty-rect flow must clear the dirty depth region to `1.0` before the prepass.
	- The Early-Z prepass, G-buffer geometry pass, and legacy color pass must clip to resolved dirty rects.

## Usage

### Backend Configuration

```ts
import { SoftwareBackend } from "../src/backends/software/SoftwareBackend";
import { WebGLBackend } from "../src/backends/webgl/WebGLBackend";
import { WebGPUBackend } from "../src/backends/webgpu/WebGPUBackend";

// SoftwareBackend setup
const softwareBackend = new SoftwareBackend({
	rasterMode: "tile",
	enableEarlyZPrepass: true,
	tile: {
		tileSize: 32,
		workerCount: 4,
	},
});

// WebGLBackend setup
const webglBackend = new WebGLBackend({
	enableEarlyZPrepass: true,
});
console.log(webglBackend.isEarlyZPrepassEnabled());

// WebGPUBackend setup
const webgpuBackend = new WebGPUBackend({
	enableEarlyZPrepass: true,
});
```

### Custom Shader Material Configuration (WebGL)

```ts
import { ShaderMaterial } from "../src/materials/ShaderMaterial";

const webglMaterial = new ShaderMaterial({
	chunks: [
		{
			backend: "webgl",
			language: "glsl",
			stage: "vertex",
			code: "#version 300 es\nvoid main() { gl_Position = vec4(0.0); }",
		},
		{
			backend: "webgl",
			language: "glsl",
			stage: "fragment-depth",
			code: "#version 300 es\nprecision highp float;\nvoid main() {}",
		},
	],
});

const depthProgram = webglMaterial.resolveWebGLDepthPrepassProgram("single");
```

### Custom Shader Material Configuration (WebGPU)

```ts
import { AlphaMode, ShaderMaterial } from "../src/materials";

const webgpuMaterial = new ShaderMaterial({
	alphaMode: AlphaMode.Mask,
	vertexEntryPoint: "vsMain",
	depthFragmentEntryPoint: "fsDepthMask",
	depthFragmentCode: `
@fragment
fn fsDepthMask() {
	// alpha discard logic
}
`,
});
```

### Verification Commands

```bash
# Software Backend tests
bun tests/static/software/test_software_early_z_prepass.mjs

# WebGL Backend tests
bun tests/static/webgl/test_webgl_backend_v2.mjs

# WebGPU Backend tests
bun tests/static/webgpu/test_webgpu_bridge.mjs
bun tests/static/webgpu/test_webgpu_frame_executor_resilience.mjs
```

## Errors & Diagnostics

- **Missing Configurations**:
	- `ShaderMaterial.resolveWebGLDepthPrepassProgram()` must return `null` when WebGL vertex or `fragment-depth` source is missing.
	- A WebGL `ShaderMaterial` using `AlphaMode.Mask` without a WebGL `fragment-depth` chunk should warn once and skip the Early-Z prepass for that material.
	- A WebGPU `ShaderMaterial` using `AlphaMode.Mask` without `depthFragmentCode` or `depthFragmentEntryPoint` must log a warning once and skip the Early-Z prepass.
- **Shader compilation failures**:
	- Custom WebGL depth prepass compile failures should throw in strict shader mode. In warn mode, the backend should warn once and skip the Early-Z prepass for that material.
	- WebGPU shader compile errors during Early-Z prepass setup should skip that material prepass path and keep color-pass rendering available.
- **Guards**: Non-finite or invalid rendering inputs must follow existing backend guards and must not introduce new crash paths.

## Compatibility / Breaking Changes

This optimization is backward compatible. Existing materials default to `depthWrite === true` and preserve prior rendering behavior.

Mask `ShaderMaterial` users must provide the appropriate backend-specific depth-discard fragments (a `fragment-depth` chunk for WebGL, or `depthFragmentCode`/`depthFragmentEntryPoint` for WebGPU) to participate in the Early-Z prepass optimization.
