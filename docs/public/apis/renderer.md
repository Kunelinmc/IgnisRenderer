# Renderer

## Scope

Use `Renderer` as the main entry point for displaying a `Scene` on an
`HTMLCanvasElement`. This guide covers the public application workflow:
choosing a backend, configuring a scene and camera, rendering frames, handling
canvas resizing and device events, and releasing resources.

The examples use only exports from the package root. Backend implementation,
frame pipeline internals, and native graphics handles are outside the scope of
this document.

## Background

A renderer needs a canvas and one rendering backend. IgnisRenderer provides
`WebGPUBackend`, `WebGLBackend`, and `SoftwareBackend`; choose the backend that
matches the environments your application supports.

Each backend instance belongs to one `Renderer` for its entire lifetime. Create
a new backend instance when creating another renderer, even after the first
renderer has been destroyed.

`Renderer` creates a default `Scene` and `Camera` when they are not supplied.
Applications can use these defaults through `renderer.scene` and
`renderer.camera`, or bind their own objects with `setScene()` and
`setCamera()`.

## API/Contract

### Creating a renderer

Construct `Renderer` with one options object:

- `canvas`: the `HTMLCanvasElement` that receives rendered output.
- `backend`: a new backend instance dedicated to this renderer.
- `camera`: an optional initial `Camera`. If it has no parent, the renderer
  adds it to the default scene.

Call `initialize()` before performing optional setup that needs an initialized
graphics device. Although the first `renderFrame()` can initialize the renderer
automatically, explicit initialization makes startup failures easier to handle.

### Scene and camera

Read the active objects from `renderer.scene` and `renderer.camera`.

Use `setScene(scene)` to switch scenes and `setCamera(camera)` to switch the
active view. The active camera must belong to the active scene. When replacing
both, add the camera to the new scene before binding the scene and camera.

Changes to the active scene normally make a future frame eligible for
rendering. Call `requestRender()` after application state changes that are not
represented by the scene, camera, textures, or other renderer-managed objects.

### Rendering frames

Choose one frame scheduling style:

- `renderLoop()` schedules serialized animation frames and returns an
  idempotent function that stops the loop. Repeated calls while the loop is
  active return the same stop function.
- `renderFrame(nowMs)` renders at most one frame and does not schedule another
  frame. Pass a timestamp from the same time base as `performance.now()` or
  `requestAnimationFrame`.

Do not call `renderFrame()` concurrently. Its result reports whether work was
rendered or skipped because an on-demand renderer was already clean:

```ts
const result = await renderer.renderFrame(performance.now());
if (!result.rendered) {
	console.log(result.reason);
}
```

`renderScene()` is a deprecated alias. New code should use `renderFrame()` or
`renderLoop()`.

### Canvas size

`initialize()` calls `resizeCanvas()` once. Call it again whenever the canvas
element's displayed size changes. It updates the drawing buffer for the current
device pixel ratio and refreshes the camera aspect ratio.

For layouts that can resize without a window resize, prefer `ResizeObserver`.

### Events and diagnostics

Subscribe with `renderer.on()` to observe commonly useful lifecycle events:

- `tick`: emitted when a frame attempt begins, with `now` and `deltaTime` in
  milliseconds.
- `framestart` and `frameend`: emitted around a rendered frame.
- `devicelost`: reports that the active graphics device or context was lost.
- `devicerestored`: reports that restoration completed.

`getBackendDebugInfo()` returns a best-effort diagnostic snapshot. Check its
`available` property before reading optional device, limit, or feature fields.
Diagnostic identifiers can be absent or redacted and should not be used for
feature decisions.

### Cleanup

Keep the stop function returned by `renderLoop()` when the application needs to
pause rendering. Call `destroy()` when the renderer is no longer needed.
Destruction stops the active loop, waits for an in-progress frame, and releases
renderer resources. A destroyed renderer cannot be reused; create a new
renderer and backend instead.

## Usage

The following example creates a renderer, uses its default scene and camera,
keeps the canvas resolution synchronized with its displayed size, and cleans
up all renderer-owned resources:

```ts
import {
	Renderer,
	WebGPUBackend,
	type RenderBackendDeviceLostInfo,
} from "ignisrenderer";

const canvas = document.querySelector<HTMLCanvasElement>("#viewport");
if (!canvas) {
	throw new Error("Canvas #viewport was not found.");
}

const renderer = new Renderer({
	canvas,
	backend: new WebGPUBackend(),
});

await renderer.initialize();

const resizeObserver = new ResizeObserver(() => {
	renderer.resizeCanvas();
});
resizeObserver.observe(canvas);

const handleDeviceLost = ({ info }: {
	info?: RenderBackendDeviceLostInfo;
}): void => {
	console.warn("Rendering device lost.", info?.message);
};
renderer.on("devicelost", handleDeviceLost);

const stopRenderLoop = renderer.renderLoop();

async function dispose(): Promise<void> {
	stopRenderLoop();
	resizeObserver.disconnect();
	renderer.off("devicelost", handleDeviceLost);
	await renderer.destroy();
}
```

To use an application-created scene and camera:

```ts
import {
	Camera,
	Renderer,
	Scene,
	WebGLBackend,
} from "ignisrenderer";

const scene = new Scene();
const camera = scene.add(new Camera({
	fov: 60,
	near: 0.1,
	far: 5000,
}));
camera.position.set(0, 2, 5);

const renderer = new Renderer({
	canvas,
	backend: new WebGLBackend(),
	camera,
});
renderer.setScene(scene);

await renderer.initialize();
await renderer.renderFrame(performance.now());
await renderer.destroy();
```

## Errors & Diagnostics

- `Renderer.initialize(): already initialized.`: call `initialize()` only once.
  Rendering methods can be used after that promise resolves.
- A method reports that it cannot run after `destroy()`: create a new
  `Renderer` with a new backend instance.
- `Renderer.renderFrame() cannot run concurrently.`: await the current frame
  before requesting another manual frame.
- Scene or camera binding reports that the camera is not in the scene: add the
  camera to that scene before calling `setScene()` or `setCamera()`.
- The output size does not match the layout: ensure the canvas has a non-zero
  displayed size and call `resizeCanvas()` after the layout changes.
- `getBackendDebugInfo().available` is `false`: initialize the renderer first.
  Some environments may still withhold optional diagnostics.
- A backend is unavailable in the current browser: select a supported backend
  and create a new renderer. Do not attach the existing backend instance to a
  replacement renderer.

## Compatibility / Breaking Changes

`Renderer` requires a single options object. The positional
`new Renderer(backend, canvas, camera)` form is not supported.

Use `renderFrame(nowMs)` for one manually scheduled frame and `renderLoop()` for
continuous rendering. `renderScene(nowMs)` remains only as a deprecated
compatibility alias.

Backend instances are single-use renderer dependencies. Code that replaces or
recreates a renderer must also construct a new backend instance.
