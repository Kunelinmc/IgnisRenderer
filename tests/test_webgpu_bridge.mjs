import assert from 'node:assert/strict'
import { ResourceManager } from '../src/core/bridge/ResourceManager.ts'
import { WEBGPU_SCENE_SHADER } from '../src/core/bridge/webgpuShaders.ts'
import {
	createWebGPUMaterialUniformData,
	packMatrix4ForWGSL,
	remapClipSpaceDepth,
	resolveWebGPUFeatureState,
	WEBGPU_FRAME_UNIFORM_FLOATS,
} from '../src/core/bridge/webgpuUtils.ts'
import { BufferUsage } from '../src/core/ral/types.ts'
import { Matrix4 } from '../src/maths/Matrix4.ts'
import { PBRMaterial } from '../src/materials/PBRMaterial.ts'
import { PhongMaterial } from '../src/materials/PhongMaterial.ts'
import { UnlitMaterial } from '../src/materials/UnlitMaterial.ts'
import { SimpleModel } from '../src/models/SimpleModel.ts'

class FakeDevice {
	constructor() {
		this.type = 'webgpu'
		this.canvasFormat = 'rgba8unorm'
		this.bufferDescs = []
	}

	async init() {}

	createBuffer(desc) {
		this.bufferDescs.push(desc)
		return {
			size: desc.size,
			desc,
			destroy() {},
		}
	}

	createTexture(desc) {
		return {
			width: desc.width,
			height: desc.height,
			desc,
			destroy() {},
		}
	}

	createSampler(desc) {
		return { label: desc.label, desc }
	}

	createShaderModule(desc) {
		return { label: desc.label, desc }
	}

	createPipeline(desc) {
		return { label: desc.label, desc }
	}

	createComputePipeline(desc) {
		return { label: desc.label, desc }
	}

	createBindingGroup(desc) {
		return { label: desc.label, desc }
	}

	createCommandEncoder() {
		throw new Error('Not needed in this test')
	}

	writeBuffer(buffer, data) {
		buffer.lastWrite = data
	}

	writeTexture(texture, data, layout, size) {
		texture.lastWrite = { data, layout, size }
	}

	copyTextureToTexture() {}
	submit() {}
	resize() {}
}

function createModel(material) {
	return new SimpleModel([
		{
			material,
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	])
}

function testMatrixPackingAndDepthRemap() {
	const matrix = new Matrix4([
		[1, 2, 3, 4],
		[5, 6, 7, 8],
		[9, 10, 11, 12],
		[13, 14, 15, 16],
	])

	assert.deepEqual(Array.from(packMatrix4ForWGSL(matrix)), [
		1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16,
	])
	assert.equal(remapClipSpaceDepth(-1, 1), 0)
	assert.equal(remapClipSpaceDepth(1, 1), 1)
}

function testMaterialAdaptation() {
	const pbr = new PBRMaterial({
		albedo: { r: 128, g: 64, b: 32 },
		roughness: 0.25,
		metalness: 0.75,
		reflectance: 0.6,
	})
	const pbrData = createWebGPUMaterialUniformData(pbr)
	assert.ok(Math.abs(pbrData.baseColorFactor[0] - 128 / 255) < 1e-6)
	assert.ok(Math.abs(pbrData.surfaceParams0[0] - 0.25) < 1e-6)
	assert.ok(Math.abs(pbrData.surfaceParams0[1] - 0.75) < 1e-6)
	assert.ok(Math.abs(pbrData.surfaceParams0[2] - 0.6) < 1e-6)
	assert.equal(pbrData.textureSlots.length, 14)

	const phong = new PhongMaterial({
		diffuse: { r: 128, g: 128, b: 128 },
		specular: { r: 255, g: 128, b: 64 },
		shininess: 24,
	})
	const phongData = createWebGPUMaterialUniformData(phong)
	assert.ok(
		phongData.baseColorFactor[0] > 0.2 && phongData.baseColorFactor[0] < 0.22
	)
	assert.equal(phongData.materialFlags[0], 0)
	assert.equal(phongData.phongAmbientShininess[3], 24)
	assert.ok(phongData.phongSpecularShading[0] > 0.9)

	const unlit = new UnlitMaterial({
		diffuse: { r: 255, g: 32, b: 16 },
	})
	const unlitData = createWebGPUMaterialUniformData(unlit)
	assert.equal(unlitData.materialFlags[0], 2)
}

function testFeatureGate() {
	const featureState = resolveWebGPUFeatureState({
		enableLighting: true,
		enableGamma: true,
		enableSH: true,
		enableShadows: true,
		enableReflection: true,
		enableSkybox: true,
		enableSSAO: true,
		enableVolumetric: true,
	})

	assert.equal(featureState.enableLighting, true)
	assert.equal(featureState.enableGamma, true)
	assert.equal(featureState.enableSH, false)
	assert.equal(featureState.enableShadows, true)
	assert.equal(featureState.enableReflection, false)
	assert.equal(featureState.enableSkybox, false)
	assert.equal(featureState.enableSSAO, false)
	assert.equal(featureState.enableVolumetric, false)
	assert.ok(featureState.warnings.length >= 5)
}

function testSceneShaderCoverage() {
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			'let pointCount = u32(frame.lightCounts.y + 0.5);'
		)
	)
	assert.ok(
		WEBGPU_SCENE_SHADER.includes(
			'frame.pointLights[i].positionRange.xyz - input.worldPosition'
		)
	)
	assert.ok(WEBGPU_SCENE_SHADER.includes('sampleDirectionalShadowVisibility'))
	assert.ok(WEBGPU_SCENE_SHADER.includes('textureLoad(directionalShadowAtlas'))
}

async function testResourceManagerUsesCopyDstForWebGPUUploads() {
	const device = new FakeDevice()
	const resourceManager = new ResourceManager(device)
	const model = createModel(
		new PBRMaterial({
			albedo: { r: 255, g: 255, b: 255 },
		})
	)

	resourceManager.getGeometry(model)
	resourceManager.prepareWebGPUFrame(
		{
			scene: { lights: [] },
			camera: {
				viewProjectionMatrix: Matrix4.identity(),
				position: { x: 0, y: 0, z: 5 },
			},
			warnOnce() {},
		},
		resolveWebGPUFeatureState({})
	)
	const draw = await resourceManager.getWebGPUDrawResources(model, {
		warnOnce() {},
	})

	assert.ok(draw)
	assert.equal(draw.frameBinding.desc.entries.length, 3)
	assert.equal(draw.modelBinding.desc.entries.length, 29)
	assert.ok(
		device.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Vertex) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	)
	assert.ok(
		device.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Index) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	)
	assert.ok(
		device.bufferDescs.some(
			(desc) =>
				(desc.usage & BufferUsage.Uniform) !== 0 &&
				(desc.usage & BufferUsage.CopyDst) !== 0
		)
	)
	assert.ok(
		device.bufferDescs.some(
			(desc) => desc.size === WEBGPU_FRAME_UNIFORM_FLOATS * 4
		)
	)
}

async function run() {
	testMatrixPackingAndDepthRemap()
	testMaterialAdaptation()
	testFeatureGate()
	testSceneShaderCoverage()
	await testResourceManagerUsesCopyDstForWebGPUUploads()
	console.log('WebGPU bridge tests passed')
}

await run()
