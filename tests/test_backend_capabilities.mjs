import assert from 'node:assert/strict'
import { SoftwareBackend } from '../src/renderers/SoftwareBackend.ts'
import { WebGPUBackend } from '../src/renderers/WebGPUBackend.ts'
import { WebGLBackend } from '../src/renderers/webgl/WebGLBackend.ts'

function run() {
	const software = new SoftwareBackend()
	const webgpu = new WebGPUBackend()
	const webgl = new WebGLBackend()

	assert.deepEqual(software.capabilities, {
		sh: true,
		shadows: true,
		reflection: true,
		skybox: true,
		ssao: true,
		taa: false,
		ssr: false,
		volumetric: true,
	});

	assert.deepEqual(webgpu.capabilities, {
		sh: true,
		shadows: true,
		reflection: false,
		skybox: true,
		ssao: true,
		taa: true,
		ssr: true,
		volumetric: true,
	});

	assert.deepEqual(webgl.capabilities, {
		sh: false,
		shadows: false,
		reflection: false,
		skybox: false,
		ssao: false,
		taa: false,
		ssr: false,
		volumetric: false,
	});

	testSoftwareBackendReusesFrameImageData()

	console.log('Backend capability tests passed')
}

function testSoftwareBackendReusesFrameImageData() {
	const OriginalImageData = globalThis.ImageData
	const created = []

	class FakeImageData {
		constructor(dataOrWidth, widthOrHeight, maybeHeight) {
			if (dataOrWidth instanceof Uint8ClampedArray) {
				this.data = dataOrWidth
				this.width = widthOrHeight
				this.height = maybeHeight
			} else {
				this.width = dataOrWidth
				this.height = widthOrHeight
				this.data = new Uint8ClampedArray(this.width * this.height * 4)
			}
			created.push(this)
		}
	}

	globalThis.ImageData = FakeImageData

	try {
		const backend = new SoftwareBackend()
		const pixels = new Uint8ClampedArray(16)
		pixels[0] = 7
		const putCalls = []

		backend._renderer = {
			pixels,
			canvas: {
				width: 2,
				height: 2,
			},
		}
		backend._ctx = {
			putImageData(imageData, x, y) {
				putCalls.push({ imageData, x, y })
			},
		}

		backend.endFrame()
		pixels[0] = 21
		backend.endFrame()

		assert.equal(created.length, 1)
		assert.equal(putCalls.length, 2)
		assert.equal(putCalls[0].x, 0)
		assert.equal(putCalls[0].y, 0)
		assert.strictEqual(putCalls[0].imageData, putCalls[1].imageData)
		assert.strictEqual(putCalls[0].imageData.data, pixels)
		assert.equal(putCalls[1].imageData.data[0], 21)
	} finally {
		globalThis.ImageData = OriginalImageData
	}
}

run()
