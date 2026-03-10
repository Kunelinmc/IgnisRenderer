import assert from 'node:assert/strict'
import { Camera } from '../src/cameras/Camera.ts'
import { Texture } from '../src/core/Texture.ts'
import { Renderer } from '../src/renderers/Renderer.ts'

class StubBackend {
	constructor() {
		this.type = 'stub'
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			skybox: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
		}
		this.frameScheduling = 'on-demand'
		this.beginFrameCount = 0
	}

	async init() {}

	resize() {}

	getAttachments(width, height) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		}
	}

	beginFrame() {
		this.beginFrameCount++
	}

	executePass() {}

	endFrame() {}
}

class FakeDynamicTexture extends Texture {
	constructor(framesToUpdate) {
		super(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1, 'sRGB')
		this._framesToUpdate = framesToUpdate
		this._registerAsDynamicTexture()
	}

	update() {
		if (this._framesToUpdate <= 0) {
			return false
		}
		this._framesToUpdate--
		this.markNeedsUpdate()
		return true
	}
}

async function run() {
	const originalWindow = globalThis.window
	const originalRAF = globalThis.requestAnimationFrame

	try {
		globalThis.window = { devicePixelRatio: 1 }
		globalThis.requestAnimationFrame = () => 0

		const backend = new StubBackend()
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 }
			},
		}
		const camera = new Camera()
		const renderer = new Renderer(backend, canvas, camera)
		const dynamicTexture = new FakeDynamicTexture(2)

		await renderer.renderScene(0)
		await renderer.renderScene(16)
		await renderer.renderScene(32)

		assert.equal(backend.beginFrameCount, 2)

		dynamicTexture.dispose()
		console.log('Renderer dynamic texture update tests passed')
	} finally {
		globalThis.window = originalWindow
		globalThis.requestAnimationFrame = originalRAF
	}
}

await run()
