import {
	AmbientLight,
	DirectionalLight,
	GLTFLoader,
	MeshAsset,
	MeshInstance,
	Node,
	OrbitCamera,
	PBRMaterial,
	ParticleBlendMode,
	ParticleSystem,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
} from './index'

interface RendererBootstrap {
	canvas: HTMLCanvasElement
	renderer: Renderer
}

async function init() {
	let canvas = document.getElementById('canvas3d') as HTMLCanvasElement
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500)
	const scene = new Scene()
	scene.add(camera)

	scene.add(
		new AmbientLight({
			color: { r: 255, g: 255, b: 255 },
			intensity: 0.5,
		})
	)

	scene.add(
		new DirectionalLight({
			color: { r: 255, g: 255, b: 255 },
			direction: { x: -1, y: -1, z: -1 },
			intensity: 2.5,
		})
	)

	const loader = new GLTFLoader()
	const gltfRoot = await loader.load('./assets/duck.glb')
	scene.add(gltfRoot)
	scaleLoadedMeshToTargetRadius(gltfRoot, scene, 120)

	scene.add(createGroundPlane())

	scene.add(
		new ParticleSystem({
			name: 'fountain',
			maxParticles: 5000,
			position: { x: -120, y: 0, z: 120 },
			emit: {
				rate: 300,
				direction: { x: 0, y: 1, z: 0 },
				spread: 0.15,
				speedRange: [60, 100],
				sizeRange: [5, 12],
				startColor: { r: 100, g: 200, b: 255, a: 0.8 },
				rotationRange: [0, Math.PI * 2],
				angularVelocityRange: [-2, 2],
			},
			gravity: { x: 0, y: -120, z: 0 },
			sizeOverLifetime: [
				{ t: 0, value: 1.0 },
				{ t: 1, value: 0.2 },
			],
			colorOverLifetime: [
				{ t: 0, value: { r: 100, g: 200, b: 255, a: 0.8 } },
				{ t: 0.8, value: { r: 150, g: 220, b: 255, a: 0.4 } },
				{ t: 1, value: { r: 200, g: 240, b: 255, a: 0 } },
			],
		})
	)

	scene.add(
		new ParticleSystem({
			name: 'fire',
			maxParticles: 2000,
			position: { x: 120, y: 10, z: -120 },
			blendMode: ParticleBlendMode.Additive,
			emit: {
				rate: 150,
				direction: { x: 0, y: 1, z: 0 },
				spread: 0.3,
				speedRange: [20, 40],
				sizeRange: [5, 12],
				startColor: { r: 255, g: 200, b: 50, a: 1 },
				rotationRange: [0, Math.PI * 2],
				angularVelocityRange: [-3, 3],
			},
			gravity: { x: 0, y: 10, z: 0 },
			sizeOverLifetime: [
				{ t: 0, value: 0.2 },
				{ t: 0.2, value: 1.0 },
				{ t: 1, value: 0.5 },
			],
			colorOverLifetime: [
				{ t: 0, value: { r: 255, g: 255, b: 200, a: 1 } },
				{ t: 0.3, value: { r: 255, g: 150, b: 0, a: 0.8 } },
				{ t: 0.6, value: { r: 200, g: 50, b: 0, a: 0.5 } },
				{ t: 1, value: { r: 50, g: 0, b: 0, a: 0 } },
			],
		})
	)

	const bootstrap = await createRenderer(canvas, camera, scene)
	canvas = bootstrap.canvas
	const renderer = bootstrap.renderer

	renderer.updateSH()
	renderer.requestRender()

	bindControls(canvas, camera, renderer)

	window.addEventListener('resize', () => {
		renderer.resizeCanvas()
		renderer.requestRender()
	})
}

function createGroundPlane(): MeshInstance {
	const size = 400
	const half = size / 2
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
		doubleSided: true,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
		reflectivity: 0.5,
	})
	const mesh = MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{ x: -half, y: 0, z: -half, u: 0, v: 0, normal: { x: 0, y: 1, z: 0 } },
				{ x: half, y: 0, z: -half, u: 1, v: 0, normal: { x: 0, y: 1, z: 0 } },
				{ x: half, y: 0, z: half, u: 1, v: 1, normal: { x: 0, y: 1, z: 0 } },
			],
		},
		{
			material,
			vertices: [
				{ x: -half, y: 0, z: -half, u: 0, v: 0, normal: { x: 0, y: 1, z: 0 } },
				{ x: half, y: 0, z: half, u: 1, v: 1, normal: { x: 0, y: 1, z: 0 } },
				{ x: -half, y: 0, z: half, u: 0, v: 1, normal: { x: 0, y: 1, z: 0 } },
			],
		},
	])
	return new MeshInstance({
		mesh,
		name: 'ground',
	})
}

function scaleLoadedMeshToTargetRadius(
	root: Node,
	scene: Scene,
	targetRadius: number
): void {
	const meshes: MeshInstance[] = []
	root.traverse((node) => {
		if (node instanceof MeshInstance) {
			meshes.push(node)
		}
	})
	const primary = meshes[0]
	if (!primary) return

	const scale = targetRadius / Math.max(primary.mesh.boundingSphere.radius, 1e-6)
	primary.scale.set(scale, scale, scale)
	primary.updateLocalMatrix()
	scene.updateWorldMatrices()
	primary.position.y = -primary.getWorldBoundingBox().min.y
}

async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	scene: Scene
): Promise<RendererBootstrap> {
	if (navigator.gpu) {
		const webgpuRenderer = new Renderer(new WebGPUBackend(), canvas, camera)
		webgpuRenderer.setScene(scene)
		configureRenderer(webgpuRenderer)

		try {
			await webgpuRenderer.init()
			console.info('Using WebGPU backend')
			return {
				canvas,
				renderer: webgpuRenderer,
			}
		} catch (error) {
			console.warn(
				'WebGPU initialization failed, falling back to software.',
				error
			)
		}
	}

	const softwareRenderer = new Renderer(new SoftwareBackend(), canvas, camera)
	softwareRenderer.setScene(scene)
	configureRenderer(softwareRenderer)
	await softwareRenderer.init()
	console.info('Using software backend')

	return {
		canvas,
		renderer: softwareRenderer,
	}
}

function configureRenderer(renderer: Renderer): void {
	if (renderer.backendType === 'webgpu') {
		renderer.features.enableSH = true
		renderer.features.enableShadows = true
		renderer.features.enableReflection = false
		renderer.features.enableSkybox = true
		renderer.features.enableSSAO = false
		renderer.features.enableTAA = false
		renderer.features.enableFXAA = true
		renderer.features.enableSSR = false
		renderer.features.enableVolumetric = false
		return
	}

	renderer.features.enableSH = true
	renderer.features.enableShadows = true
	renderer.features.enableReflection = true
}

function bindControls(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	renderer: Renderer
): void {
	let isDragging = false
	let lastMouse = { x: 0, y: 0 }

	canvas.addEventListener('mousedown', (event) => {
		isDragging = true
		lastMouse = { x: event.clientX, y: event.clientY }
	})

	window.addEventListener('mousemove', (event) => {
		if (!isDragging) return
		camera.rotate(event.clientX - lastMouse.x, event.clientY - lastMouse.y)
		lastMouse = { x: event.clientX, y: event.clientY }
		renderer.requestRender()
	})

	window.addEventListener('mouseup', () => {
		isDragging = false
	})

	canvas.addEventListener(
		'wheel',
		(event) => {
			event.preventDefault()
			camera.zoom(event.deltaY)
			renderer.requestRender()
		},
		{ passive: false }
	)

	canvas.addEventListener(
		'touchstart',
		(event) => {
			if (event.touches.length !== 1) return
			isDragging = true
			lastMouse = {
				x: event.touches[0].clientX,
				y: event.touches[0].clientY,
			}
		},
		{ passive: false }
	)

	canvas.addEventListener(
		'touchmove',
		(event) => {
			if (!isDragging || event.touches.length !== 1) return
			const touch = event.touches[0]
			camera.rotate(touch.clientX - lastMouse.x, touch.clientY - lastMouse.y)
			lastMouse = { x: touch.clientX, y: touch.clientY }
			renderer.requestRender()
		},
		{ passive: false }
	)

	canvas.addEventListener('touchend', () => {
		isDragging = false
	})
}

init().catch((error) => {
	console.error('Failed to initialize scene:', error)
})
