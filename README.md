# IgnisRenderer

![Version](https://img.shields.io/badge/version-1.0.0-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?logo=typescript)
![Vite](https://img.shields.io/badge/Vite-7.x-646CFF?logo=vite)

IgnisRenderer is a TypeScript 3D renderer with dual backends:

- **SoftwareBackend**: CPU rasterization pipeline
- **WebGPUBackend**: hardware-accelerated pipeline

The current major architecture is a **Scene Graph** with
`Node` + `MeshAsset`/`MeshInstance`.

[Live Demo](https://ignis-renderer-demo.netlify.app/)

![IgnisRenderer Screenshot](./assets/screenshot.png)

## Highlights

- Scene graph transforms with parent-child hierarchy
- Shared mesh resources via `MeshAsset`
- Per-node mesh placement via `MeshInstance`
- Lights, cameras, and particle systems as scene nodes
- glTF/GLB and OBJ loading into scene graph nodes
- Software and WebGPU pipelines using the same prepared scene contracts

## Scene Graph Model

- `Node`: transform + hierarchy (`position`, `quaternion`, `scale`,
  `localMatrix`, `worldMatrix`)
- `MeshAsset`: primitives and local bounds (resource)
- `MeshInstance extends Node`: references one `MeshAsset`
- `Light extends Node`
- `Camera extends Node`
- `ParticleSystem extends Node`

`Scene` API:

- `scene.add(node)`
- `scene.remove(node)`
- `scene.traverse(visitor)`
- `scene.contains(node)`
- `scene.getMeshInstances()/getLights()/getCameras()/getParticleSystems()`

## Quick Start

```typescript
import {
	AmbientLight,
	DirectionalLight,
	GLTFLoader,
	MeshFactory,
	OrbitCamera,
	PBRMaterial,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
} from 'ignisrenderer'

async function main(canvas: HTMLCanvasElement) {
	const scene = new Scene()
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500)
	scene.add(camera)

	scene.add(
		new AmbientLight({
			color: { r: 255, g: 255, b: 255 },
			intensity: 0.3,
		})
	)

	scene.add(
		new DirectionalLight({
			direction: { x: -1, y: -1, z: -1 },
			intensity: 2,
		})
	)

	const loader = new GLTFLoader()
	const gltfRoot = await loader.load('./assets/duck.glb')
	scene.add(gltfRoot)

	const ground = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		400,
		400,
		new PBRMaterial({ albedo: { r: 255, g: 255, b: 255 } })
	)
	scene.add(ground)

	const backend = navigator.gpu ? new WebGPUBackend() : new SoftwareBackend()
	const renderer = new Renderer(backend, canvas, camera)

	// Camera must belong to the active scene graph.
	renderer.setScene(scene)
	renderer.setCamera(camera)

	await renderer.init()
}
```

## Loader Behavior

- `GLTFLoader` / `GLBLoader` return a `Node` root.
- glTF hierarchy is preserved.
- Node transforms are not baked into mesh vertices.
- glTF cameras and `KHR_lights_punctual` are parsed as scene nodes.
- `OBJLoader` returns a root node containing mesh instances.

## Breaking Changes (Scene Graph Release)

- Removed old model API:
  - `SimpleModel`
  - `IModel`
  - `ModelFactory`
- Removed split scene insertion API:
  - `scene.addModel(...)`
  - `scene.addLight(...)`
  - `scene.addParticleSystem(...)`
- Use `scene.add(node)` for all node types.
- `DirectionalLight`/`SpotLight` use `direction` in local space.
- Rendering packets are instance-oriented (`meshInstance + meshAsset`).

## Project Structure

- `src/core/`: `Node`, `Scene`, shared runtime primitives
- `src/meshes/`: `MeshAsset`, `MeshInstance`, mesh construction helpers
- `src/pipeline/`: feature resolution, frame planning, prepared scene building
- `src/renderers/`: backend interface and software/webgpu implementations
- `src/shaders/`: software shader logic and WGSL modules
- `src/loaders/`: glTF/GLB/OBJ/texture/HDR loaders
- `src/maths/`: vectors, matrices, quaternions, frustum and SH math

## Development

```bash
npm install
npm run dev
```

Run tests:

```bash
npm test
```

Targeted suites:

```bash
npm run test:lighting
npm run test:pointspot
npm run test:sh
npm run test:winding
npm run test:sparse
```

## License

MIT. See [LICENSE](LICENSE).
