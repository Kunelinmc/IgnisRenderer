# Post-Process Backend Execution Contract

## Scope

This document defines the backend-specific behavior, context interfaces, and
execution flow for post-processing across Software, WebGL, and WebGPU backends.
Shared logical analysis is defined by
`docs/rendergraph/internal-render-graph-architecture.md`.

## Background

To support decoupled backends, post-process execution is delegated to backend-owned runtimes. Each backend instantiates its own `BackendPostProcessRuntime` and executes it as a `"postprocess"` pass. Execution models differ per backend: WebGPU utilizes compute shaders, WebGL uses fragment shader passes on fullscreen triangles, and Software performs CPU-based direct pixel modification.

## API/Contract

### Common Backend Lifecycle
- Backends that support post-processing must set `BackendCapabilities.postProcess = true` and handle `executePass({ stage: "postprocess" }, context)`.
- A post-process implementation must return `preservesOutsideDirtyTiles: true`
  only when it preserves all pixels outside the current dirty tiles. Omitted
  declarations must force the backend's completed frame coverage to
  `"full-frame"`.
- Backend resize, device loss, or destruction must trigger post-process runtime invalidation or destruction.
- The runtime must compile pass eligibility before it prepares pool resources,
  execute the resulting logical subgraph within the existing single
  `"postprocess"` stage, and commit history only after the enclosing backend
  frame succeeds.
- The logical post-process subgraph must continue using the pure shared
  `RenderGraphCompiler`, but it must remain nested behind one outer backend
  frame-graph node in V1.
- Shared shadow diagnostics must not fail post-process execution. Only enforced
  error diagnostics may reject the logical subgraph.
- Empty or clean incremental post-process frames must not begin a post-process
	graph transaction or prepare post-process resources.
- The runtime must retain planned and resolved logical color identities. A
	  skipped pass must alias its planned output to its resolved input.
- A custom post-process pass must include its respective entry in `PostProcessPassConfig.implementations` (e.g., `webgpu`, `webgl`, or `software`).
- `PostProcessHistoryDescriptor`, `PostProcessTransientDescriptor`, and
  `PostProcessResourceDescriptor` must share
  `PostProcessBaseResourceDescriptor` fields for `id`, `format`, and `usage`.
  Scale-based descriptors must use `PostProcessScaledResourceDescriptor`;
  concrete backend descriptors must use absolute `width` and `height`.
- `LogicalGBufferSemantic` must include `roughness`, `metallic`, and
  `specular` material channels. Backends must expose these semantics only when
  the current frame has concrete resources for the values.
- Multiple logical semantics may reference the same backend resource when the
  values are packed into different channels. The `encoding` field must identify
  the channel layout, such as `normal-roughness-metallic.z`.

---

### WebGPU Execution Contract
- `WebGPUBackend` executes the post-process graph via compute shaders.
- `WebGPUBackend` must create and destroy its `BackendPostProcessRuntime` with
  the active device session. The runtime must be destroyed before frame and
  shared device resources during device loss or backend teardown.
- `WebGPUPostProcessExecutor` must use a frame-scoped
  `WebGPUPostProcessSessionPort` for G-buffer, execution-context, completed-pass,
  binding-invalidation, and presentation operations. It must not resolve a
  `WebGPUFrameOrchestrator` back-reference.
- The context passed to WebGPU implementations is typed as `WebGPURuntimePostProcessContext` (or `WebGPUScreenPostProcessContext`), providing `encoder`, `targets`, and `shared` (`PostProcessSharedContext`).
- Passes must not mutate the frame targets directly; they must publish their output color texture via the `publishColorTarget(texture)` callback.
- The backend must treat `publishColorTarget(texture)` as pending until the
  pass reports success. It must reject a controlled publication followed by
  `{ ran: false }`.
- Warmup planning collects ordered descriptors and runs `PostProcessPassImplementation.warmup(context)` if present.
- WebGPU passes requesting temporary resources must declare them using `getTransientResourceDescriptors(request)`. The runtime injects the allocated resources under the properties defined in `metadata.context.transients`.
- Built-in WebGPU passes that consume the shared opaque-depth Hi-Z must mark
  their graph metadata with a required `backend:frame-hiz` backend-shared
  resource and their internal context metadata with `requiresHiZ`. WebGPU must
  report that resource as available only after the frame graph has built Hi-Z
  successfully; it is not a post-process transient resource.
- WebGPU must expose `roughness` and `metallic` logical channels from
  `gNormalRoughMetal.z` and `gNormalRoughMetal.w` when `gNormalRoughMetal`
  exists.
- WebGPU must expose the `specular` logical channel from `gSpecular.rgba` when
  `gSpecular` exists.

---

### WebGL Execution Contract
- `WebGLBackend` executes post-processing by drawing fullscreen triangles using fragment shaders into ping-pong framebuffers.
- When the runtime supports at least five draw buffers and color attachments,
  WebGL must expose `albedo`, `roughness`, `metallic`, and `specular` from its
  material G-buffer when an active WebGL implementation requires them. The
  channels must report their concrete runtime formats and the encodings
  `linear-rgb-alpha`, `normal-roughness-metallic.z`,
  `normal-roughness-metallic.w`, and `specular-color-factor.rgba`.
- When the five-target requirement is unavailable, WebGL must omit these
  material channels while preserving available legacy channels. The graph must
  skip a pass whose declared requirements are no longer satisfied.
- The context passed is typed as `WebGLScreenPostProcessContext`, providing access to `gl`, `programCompiler`, `fullscreenVao`, `postFramebuffer`, `sceneColorTexture`, and texture dimensions.
- WebGL implementations must use the following helper methods on the context:
  - `getSourceTexture()`: Retrieve the source texture to sample.
  - `resolveTargetTexture(sourceTexture)`: Resolve the appropriate output texture.
  - `bindColorTarget(texture)`: Bind the target texture as the framebuffer attachment.
  - `drawFullscreen()`: Draw the fullscreen triangle.
  - `publishColorTexture(texture)`: Notify the runtime of the final written texture.
- WebGL must treat `publishColorTexture(texture)` as pending until the pass
  result is known. Presentation must read the last successfully resolved
	color texture, not a planned ping-pong target.
- A built-in WebGL pass must publish the target selected for its logical node;
	  publishing another owned texture must fail the frame transaction.
- WebGL built-in passes, including `gamma`, must execute through
  `WebGLScreenPostProcessContext` instead of presenting directly to the canvas.
- Pass program compilation is managed via `WebGLProgramCompiler` slots, which handle validation, uniform reflection, and warmup.

---

### Software Execution Contract
- `SoftwareBackend` executes post-processing via CPU-based direct buffer manipulations.
- The context passed is `SoftwareBuiltinPostProcessContext`, containing `canvasContext`.
- Passes must fetch the target pixel array from the frame attachments (`request.frameContext.attachments.pixels`).
- To optimize performance, Software passes should resolve dirty rectangles using `resolveSoftwareDirtyRects(request.frameContext)` and process pixels only within these bounding regions.

## Usage


### WebGPU Implementation Example
```ts
import { PostProcessPass, type PostProcessPassResolveRequest, type PostProcessTransientDescriptor } from "ignisrenderer";

class CustomWebGPUPass extends PostProcessPass {
	constructor() {
		super({
			id: "custom-webgpu",
			placement: "ldr",
			order: 5,
			enabled: true,
			implementations: {
				webgpu: {
					id: "custom-webgpu:webgpu",
					metadata: {
						context: {
							backend: "webgpu",
							kind: "screen",
							publishColorTarget: true,
							transients: [{
								property: "scratch",
								transientId: "custom-webgpu:scratch",
							}],
						},
					},
					execute(_request, context) {
						// Record WebGPU compute pass command encoder dispatches...
						return { ran: true };
					},
				},
			},
		});
	}

	public override getTransientResourceDescriptors(
		_request: PostProcessPassResolveRequest
	): readonly PostProcessTransientDescriptor[] {
		return [{
			id: "custom-webgpu:scratch",
			format: "rgba16float",
			usage: ["sampled", "storage", "render-target"],
		}];
	}
}
```

### WebGL Implementation Example
```ts
import { PostProcessPass, type PostProcessPassRequest, type PostProcessPassResult } from "ignisrenderer";
import { type WebGLScreenPostProcessContext, resolveWebGLTarget, bindWebGLPostTarget } from "ignisrenderer/passes/ScreenPassShared";

class CustomWebGLImpl {
	public readonly id = "custom-pass:webgl";
	public readonly metadata = { context: { backend: "webgl", kind: "screen" } };

	public execute(request: PostProcessPassRequest, context: WebGLScreenPostProcessContext): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) return { ran: false };
		
		const gl = context.gl;
		// Bind target, set textures, uniforms, and draw
		bindWebGLPostTarget(context, program, target.texture);
		context.drawFullscreen();
		context.publishColorTexture(target.texture);
		
		return { ran: true };
	}
}
```

### Software Implementation Example
```ts
import { PostProcessPass, type PostProcessPassRequest, type PostProcessPassResult } from "ignisrenderer";
import { resolveSoftwareDirtyRects, forEachSoftwareDirtyRect } from "ignisrenderer/passes/ScreenPassShared";

class CustomSoftwareImpl {
	public readonly id = "custom-pass:software";

	public execute(request: PostProcessPassRequest): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels) return { ran: false };

		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		const width = request.frameContext.attachments.width;

		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const idx = (row + x) << 2;
					// Modify pixel RGBA values directly in pixels buffer...
				}
			}
		});
		return { ran: true };
	}
}
```

## Errors & Diagnostics

- `postprocess-requirement-missing-<passId>`: Triggered when a backend G-buffer bridge lacks a required semantic channel (e.g. `depth`, `motion`, `roughness`) during execution.
- `postprocess-transient-conflict-<transientId>`: Triggered when eligible passes request incompatible transient descriptors for the same transient resource.
- WebGPU device allocation failures during `createResource(desc)` must propagate as backend resource allocation errors.
- WebGL float color attachment fallbacks: If float textures are requested but `EXT_color_buffer_float` is unsupported, the WebGL runtime falls back to `rgba8unorm` and triggers `webgl-hdr-float-unsupported`.
- The enclosing backend graph analysis must mark post-process coverage as
  `"coarse"`. It must not claim per-pass whole-frame visibility until the
  nested subgraph is explicitly integrated through imports and exports.

## Compatibility / Breaking Changes

- `renderer.postprocess` is no longer a WebGPU-specific extension; post-processing is fully cross-backend.
- Custom passes must migrate from WebGPU-only configurations to the cross-backend `PostProcessPass` structure utilizing backend-specific context factories.
- All temporary textures in WebGPU must be declared in `getTransientResourceDescriptors` and accessed via context bindings rather than querying the `WebGPUFrameTargets` directly.
- Shared graph analysis must not change pass eligibility, ordering, planned or
  resolved color versions, `{ ran: false }` aliasing, transient ownership, or
  history commit and abort behavior.
