# IBL Prefilter and WebGL Context Work Queue Migration

## Scope

This migration describes the breaking `IBLPrefilter` source API and the WebGL
custom render-target readback scheduling change.

## Background

WebGL frame passes, fragment IBL prefiltering, and render-target readback share
one stateful WebGL context. `WebGLContextWorkQueue` now serializes those users
and binds work to the current context generation only when execution begins.
`IBLPrefilter` now separates construction-time backend service options from
per-call execution settings.

## API/Contract

`IBLPrefilter` must receive an `IBLPrefilterServiceOptions` object. The object
may provide one `backend`. WebGPU and WebGL acceleration must resolve through
the backend's generic `IBL_PREFILTER_EXECUTOR_EXTENSION`; direct WebGPU compute
sources are removed.

`prefilterEnvironmentIBL` must receive construction settings through
`options.service`. `IBLPrefilterOptions` must contain execution settings only.

WebGL custom render-target `readColor()` must reject while a frame is active.
Idle readback must serialize with frames, and pending readback must reject on
context loss because framebuffer contents cannot cross context generations.

## Usage

Replace positional construction:

```ts
const prefilter = new IBLPrefilter({ backend });
const result = await prefilter.prefilter(environmentTexture, {
	acceleration: "auto",
});
```

Move helper sources under `source`:

```ts
const result = await prefilterEnvironmentIBL(environmentTexture, {
	service: {
		backend,
	},
	acceleration: "auto",
});
```

Schedule custom render-target readback after the frame has completed:

```ts
await renderer.render();
const pixels = await renderTarget.readColor(0);
```

## Errors & Diagnostics

WebGL context scheduling failures use internal `WebGLContextWorkError` values
with stable codes: `not-initialized`, `active-frame`, `active-pass`,
`context-lost`, and `destroyed`. Callers should use cancellation to stop an
explicit WebGL IBL request that is waiting for context restoration.

An explicit WebGL request already executing at context loss must reject and
must not replay. `auto` selection must fall back to Worker or CPU immediately
when WebGL is temporarily unavailable.

## Compatibility / Breaking Changes

This migration is breaking. `IBLPrefilterBackendSource`,
`IBLPrefilterSourceOptions`, positional constructor sources,
`IBLPrefilterOptions.backend`, `IBLPrefilterOptions.computeSource`, and direct
WebGPU compute sources are removed. Helper construction settings move from
`source` to `service`.

Readback previously performed during an active frame is no longer accepted.
Consumers must wait for frame completion before reading a WebGL custom render
target.
