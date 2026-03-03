import { sRGBToLinear } from '../../maths/Common'
import { Matrix4 } from '../../maths/Matrix4'
import type { Matrix3Arr, IVector3 } from '../../maths/types'
import { LightType, type SceneLight } from '../../lights'
import type { Material } from '../../materials/Material'
import type { Texture } from '../Texture'
import type { ShadowMap } from '../../utils/ShadowMapping'

export const WEBGPU_MAX_DIRECTIONAL_LIGHTS = 4
export const WEBGPU_MAX_POINT_LIGHTS = 4
export const WEBGPU_MAX_SPOT_LIGHTS = 4
export const WEBGPU_TEXTURE_SLOT_COUNT = 14
export const WEBGPU_FRAME_UNIFORM_FLOATS = 336
export const WEBGPU_MODEL_UNIFORM_FLOATS = 192

export const WEBGPU_TEXTURE_SLOT = {
	BASE_COLOR: 0,
	METALLIC_ROUGHNESS: 1,
	NORMAL: 2,
	EMISSIVE: 3,
	OCCLUSION: 4,
	SPECULAR: 5,
	SPECULAR_COLOR: 6,
	CLEARCOAT: 7,
	CLEARCOAT_ROUGHNESS: 8,
	CLEARCOAT_NORMAL: 9,
	SHEEN_COLOR: 10,
	SHEEN_ROUGHNESS: 11,
	TRANSMISSION: 12,
	THICKNESS: 13,
} as const

export interface WebGPUWarning {
	key: string
	message: string
}

export interface WebGPUFeatureState {
	enableLighting: boolean
	enableGamma: boolean
	enableSH: boolean
	enableShadows: boolean
	enableReflection: boolean
	enableSkybox: boolean
	enableSSAO: boolean
	enableVolumetric: boolean
	warnings: WebGPUWarning[]
}

export interface WebGPUDirectionalLight {
	direction: [number, number, number]
	color: [number, number, number]
}

export interface WebGPUPointLight {
	position: [number, number, number]
	range: number
	color: [number, number, number]
}

export interface WebGPUSpotLight {
	position: [number, number, number]
	range: number
	direction: [number, number, number]
	outerCos: number
	innerCos: number
	color: [number, number, number]
}

export interface WebGPUShadowData {
	enabled: boolean
	viewProjectionMatrix: Matrix4 | null
	depthBias: number
	normalBias: number
	normalBiasMin: number
	pcfRadius: number
	shadowStrength: number
	shadowMapSize: number
	atlasTileSize: number
	shadowMap: ShadowMap | null
}

export interface WebGPULightingState {
	ambientColor: [number, number, number]
	directionalLights: WebGPUDirectionalLight[]
	directionalShadows: WebGPUShadowData[]
	pointLights: WebGPUPointLight[]
	spotLights: WebGPUSpotLight[]
	spotShadows: WebGPUShadowData[]
	warnings: WebGPUWarning[]
}

export interface WebGPUTextureSlotData {
	map: Texture | null
	transformA: [number, number, number, number]
	transformB: [number, number, number, number]
}

export interface WebGPUMaterialUniformData {
	baseColorFactor: [number, number, number, number]
	emissiveFactor: [number, number, number, number]
	surfaceParams0: [number, number, number, number]
	surfaceParams1: [number, number, number, number]
	surfaceParams2: [number, number, number, number]
	surfaceParams3: [number, number, number, number]
	specularColorFactor: [number, number, number, number]
	phongAmbientShininess: [number, number, number, number]
	phongSpecularShading: [number, number, number, number]
	sheenColorClearcoatNormalScale: [number, number, number, number]
	attenuationColor: [number, number, number, number]
	materialFlags: [number, number, number, number]
	textureSlots: WebGPUTextureSlotData[]
	pipelineKey: string
	warnings: WebGPUWarning[]
}

export interface WebGPUFrameUniformInput {
	viewProjectionMatrix: Matrix4 | number[][]
	cameraPosition: IVector3
	ambientColor: [number, number, number]
	directionalLights: WebGPUDirectionalLight[]
	directionalShadows: WebGPUShadowData[]
	pointLights: WebGPUPointLight[]
	spotLights: WebGPUSpotLight[]
	spotShadows: WebGPUShadowData[]
	enableLighting: boolean
	enableGamma: boolean
	enableShadows: boolean
}

const SUPPORTED_WEBGPU_EFFECTS = {
	sh: false,
	shadows: true,
	reflection: false,
	skybox: false,
	ssao: false,
	volumetric: false,
}

export function resolveWebGPUFeatureState(params: {
	enableLighting?: boolean
	enableGamma?: boolean
	enableSH?: boolean
	enableShadows?: boolean
	enableReflection?: boolean
	enableSkybox?: boolean
	enableSSAO?: boolean
	enableVolumetric?: boolean
}): WebGPUFeatureState {
	const warnings: WebGPUWarning[] = []

	if (params.enableSH && !SUPPORTED_WEBGPU_EFFECTS.sh) {
		warnings.push({
			key: 'webgpu-feature-sh',
			message: 'WebGPU backend does not support spherical harmonics yet; disabling SH',
		})
	}

	if (params.enableShadows && !SUPPORTED_WEBGPU_EFFECTS.shadows) {
		warnings.push({
			key: 'webgpu-feature-shadows',
			message: 'WebGPU backend does not support shadows yet; disabling shadow pass',
		})
	}

	if (params.enableReflection && !SUPPORTED_WEBGPU_EFFECTS.reflection) {
		warnings.push({
			key: 'webgpu-feature-reflection',
			message:
				'WebGPU backend does not support planar reflections yet; rendering reflective materials as regular lit surfaces',
		})
	}

	if (params.enableSkybox && !SUPPORTED_WEBGPU_EFFECTS.skybox) {
		warnings.push({
			key: 'webgpu-feature-skybox',
			message: 'WebGPU backend does not support skybox rendering yet; disabling skybox',
		})
	}

	if (params.enableSSAO && !SUPPORTED_WEBGPU_EFFECTS.ssao) {
		warnings.push({
			key: 'webgpu-feature-ssao',
			message: 'WebGPU backend does not support SSAO yet; disabling SSAO',
		})
	}

	if (params.enableVolumetric && !SUPPORTED_WEBGPU_EFFECTS.volumetric) {
		warnings.push({
			key: 'webgpu-feature-volumetric',
			message: 'WebGPU backend does not support volumetric effects yet; disabling volumetrics',
		})
	}

	return {
		enableLighting: params.enableLighting !== false,
		enableGamma: params.enableGamma !== false,
		enableSH: false,
		enableShadows: params.enableShadows !== false,
		enableReflection: false,
		enableSkybox: false,
		enableSSAO: false,
		enableVolumetric: false,
		warnings,
	}
}

export function collectWebGPULighting(
	lights: SceneLight[],
	enableLighting: boolean,
	enableShadows: boolean = false,
	shadowMaps?: ReadonlyMap<any, ShadowMap>
): WebGPULightingState {
	const ambientColor: [number, number, number] = [0, 0, 0]
	const directionalLights: WebGPUDirectionalLight[] = []
	const directionalShadows: WebGPUShadowData[] = []
	const pointLights: WebGPUPointLight[] = []
	const spotLights: WebGPUSpotLight[] = []
	const spotShadows: WebGPUShadowData[] = []
	const warnings: WebGPUWarning[] = []

	if (!enableLighting) {
		return {
			ambientColor,
			directionalLights,
			directionalShadows,
			pointLights,
			spotLights,
			spotShadows,
			warnings,
		}
	}

	for (const light of lights) {
		if (light.type === LightType.Ambient) {
			ambientColor[0] += sRGBToLinear(light.color.r / 255) * light.intensity
			ambientColor[1] += sRGBToLinear(light.color.g / 255) * light.intensity
			ambientColor[2] += sRGBToLinear(light.color.b / 255) * light.intensity
			continue
		}

		if (light.type === LightType.Directional) {
			if (directionalLights.length >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) {
				warnings.push({
					key: 'webgpu-directional-limit',
					message: `WebGPU backend supports at most ${WEBGPU_MAX_DIRECTIONAL_LIGHTS} directional lights; extra lights are ignored`,
				})
				continue
			}

			const dir = normalizeVec3(
				Matrix4.transformDirection(light.worldMatrix, (light as any).dir)
			)
			directionalLights.push({
				direction: [-dir.x, -dir.y, -dir.z],
				color: toLinearLightColor(light.color, light.intensity),
			})
			directionalShadows.push(
				resolveWebGPUShadowData(
					enableShadows,
					shadowMaps?.get(light)
				)
			)
			continue
		}

		if (light.type === LightType.Point) {
			if (pointLights.length >= WEBGPU_MAX_POINT_LIGHTS) {
				warnings.push({
					key: 'webgpu-point-limit',
					message: `WebGPU backend supports at most ${WEBGPU_MAX_POINT_LIGHTS} point lights; extra lights are ignored`,
				})
				continue
			}

			const position = Matrix4.transformPoint(
				light.worldMatrix,
				(light as any).position
			)
			pointLights.push({
				position: [position.x, position.y, position.z],
				range: Math.max((light as any).range ?? 0, 0.001),
				color: toLinearLightColor(light.color, light.intensity),
			})
			continue
		}

		if (light.type === LightType.Spot) {
			if (spotLights.length >= WEBGPU_MAX_SPOT_LIGHTS) {
				warnings.push({
					key: 'webgpu-spot-limit',
					message: `WebGPU backend supports at most ${WEBGPU_MAX_SPOT_LIGHTS} spot lights; extra lights are ignored`,
				})
				continue
			}

			const position = Matrix4.transformPoint(
				light.worldMatrix,
				(light as any).position
			)
			const direction = normalizeVec3(
				Matrix4.transformDirection(light.worldMatrix, (light as any).dir)
			)
			const outerAngle = (light as any).angle ?? Math.PI / 4
			const innerAngle =
				(light as any).innerAngle ??
				outerAngle * (1 - ((light as any).penumbra ?? 0))

			spotLights.push({
				position: [position.x, position.y, position.z],
				range: Math.max((light as any).range ?? 0, 0.001),
				direction: [direction.x, direction.y, direction.z],
				outerCos: Math.cos(outerAngle),
				innerCos: Math.cos(innerAngle),
				color: toLinearLightColor(light.color, light.intensity),
			})
			spotShadows.push(
				resolveWebGPUShadowData(
					enableShadows,
					shadowMaps?.get(light)
				)
			)
			continue
		}

		warnings.push({
			key: `webgpu-light-${light.type}`,
			message: `WebGPU backend does not support ${light.type} lights yet; ignoring them for now`,
		})
	}

	return {
		ambientColor,
		directionalLights,
		directionalShadows,
		pointLights,
		spotLights,
		spotShadows,
		warnings,
	}
}

export function createWebGPUMaterialUniformData(
	material: Material
): WebGPUMaterialUniformData {
	const warnings: WebGPUWarning[] = []
	const mat = material as any
	const shadingMode = resolveShadingMode(material)
	const isPBR = shadingMode === 1
	const baseColor = getMaterialBaseColor(material, isPBR)
	const emissive = getMaterialEmissive(material, isPBR)
	const opacity = clamp(material.opacity ?? 1, 0, 1)
	const alphaMode = material.alphaMode ?? 'OPAQUE'
	const alphaModeMask = alphaMode === 'MASK' ? 1 : 0
	const alphaCutoff = clamp(material.alphaCutoff ?? 0.5, 0, 1)

	const roughness = clamp(mat.roughness ?? 0.5, 0.04, 1)
	const metalness = clamp(mat.metalness ?? 0, 0, 1)
	const reflectance = clamp(mat.reflectance ?? 0.5, 0, 1)
	const occlusionStrength = clamp(mat.occlusionStrength ?? 1, 0, 1)
	const normalScale = Math.max(0, mat.normalScale ?? 1)
	const clearcoat = clamp(mat.clearcoat ?? 0, 0, 1)
	const clearcoatRoughness = clamp(mat.clearcoatRoughness ?? 0.01, 0.04, 1)
	const sheenRoughness = clamp(mat.sheenRoughnessFactor ?? 0, 0, 1)
	const transmission = clamp(mat.transmissionFactor ?? 0, 0, 1)
	const ior = Math.max(1, mat.ior ?? 1.5)
	const thickness = Math.max(0, mat.thicknessFactor ?? 0)
	const attenuationDistance = Number.isFinite(mat.attenuationDistance)
		? Math.max(mat.attenuationDistance, 0)
		: -1
	const specularFactor = clamp(mat.specularFactor ?? 1, 0, 1)
	const clearcoatNormalScale = Math.max(0, mat.clearcoatNormalScale ?? 1)

	const specularColor = getPBRLinearColor(
		mat.specularColor ?? { r: 255, g: 255, b: 255 }
	)
	const sheenColor = getPBRLinearColor(
		mat.sheenColorFactor ?? { r: 0, g: 0, b: 0 }
	)
	const attenuationColor = getPBRLinearColor(
		mat.attenuationColor ?? { r: 255, g: 255, b: 255 }
	)
	const phongAmbient = getPhongLinearColor(
		mat.ambient ?? mat.diffuse ?? { r: 255, g: 255, b: 255 }
	)
	const phongSpecular = getPhongLinearColor(
		mat.specular ?? { r: 255, g: 255, b: 255 }
	)
	const phongShininess = Math.max(mat.shininess ?? 32, 0)
	const emissiveIntensity = clamp(mat.emissiveIntensity ?? 1, 0, 64)
	const textureSlots = createMaterialTextureSlots(material)

	pushMaterialWarnings(material, warnings)

	return {
		baseColorFactor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissiveFactor: [
			emissive[0],
			emissive[1],
			emissive[2],
			emissiveIntensity,
		],
		surfaceParams0: [roughness, metalness, reflectance, alphaCutoff],
		surfaceParams1: [
			occlusionStrength,
			normalScale,
			clearcoat,
			clearcoatRoughness,
		],
		surfaceParams2: [sheenRoughness, transmission, ior, thickness],
		surfaceParams3: [attenuationDistance, 0, 0, 0],
		specularColorFactor: [
			specularColor[0],
			specularColor[1],
			specularColor[2],
			specularFactor,
		],
		phongAmbientShininess: [
			phongAmbient[0],
			phongAmbient[1],
			phongAmbient[2],
			phongShininess,
		],
		phongSpecularShading: [
			phongSpecular[0],
			phongSpecular[1],
			phongSpecular[2],
			shadingMode,
		],
		sheenColorClearcoatNormalScale: [
			sheenColor[0],
			sheenColor[1],
			sheenColor[2],
			clearcoatNormalScale,
		],
		attenuationColor: [
			attenuationColor[0],
			attenuationColor[1],
			attenuationColor[2],
			1,
		],
		materialFlags: [
			shadingMode,
			alphaModeMask,
			material.doubleSided ? 1 : 0,
			0,
		],
		textureSlots,
		pipelineKey: [
			material.doubleSided ? 'double' : 'single',
			alphaModeMask ? 'mask' : alphaMode === 'BLEND' ? 'blend' : 'opaque',
		].join(':'),
		warnings,
	}
}

export function packMatrix4ForWGSL(
	matrix: Matrix4 | number[][]
): Float32Array {
	const elements = matrix instanceof Matrix4 ? matrix.elements : matrix

	return new Float32Array([
		elements[0][0],
		elements[1][0],
		elements[2][0],
		elements[3][0],
		elements[0][1],
		elements[1][1],
		elements[2][1],
		elements[3][1],
		elements[0][2],
		elements[1][2],
		elements[2][2],
		elements[3][2],
		elements[0][3],
		elements[1][3],
		elements[2][3],
		elements[3][3],
	])
}

export function packNormalMatrix4ForWGSL(
	normalMatrix: Matrix3Arr | Matrix4
): Float32Array {
	const rows =
		normalMatrix instanceof Matrix4 ? normalMatrix.elements : normalMatrix

	return packMatrix4ForWGSL([
		[rows[0][0], rows[0][1], rows[0][2], 0],
		[rows[1][0], rows[1][1], rows[1][2], 0],
		[rows[2][0], rows[2][1], rows[2][2], 0],
		[0, 0, 0, 1],
	])
}

export function packFrameUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	const data = new Float32Array(WEBGPU_FRAME_UNIFORM_FLOATS)
	const viewProjection = packMatrix4ForWGSL(input.viewProjectionMatrix)

	data.set(viewProjection, 0)
	data.set(
		[
			input.cameraPosition.x,
			input.cameraPosition.y,
			input.cameraPosition.z,
			1,
		],
		16
	)
	data.set(
		[
			input.ambientColor[0],
			input.ambientColor[1],
			input.ambientColor[2],
			1,
		],
		20
	)
	data.set(
		[
			input.directionalLights.length,
			input.pointLights.length,
			input.spotLights.length,
			0,
		],
		24
	)
	data.set(
		[
			input.enableLighting ? 1 : 0,
			input.enableGamma ? 1 : 0,
			input.enableShadows ? 1 : 0,
			0,
		],
		28
	)

	let offset = 32
	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		const light = input.directionalLights[i]
		if (light) {
			data.set(
				[light.direction[0], light.direction[1], light.direction[2], 0],
				offset
			)
			data.set([light.color[0], light.color[1], light.color[2], 0], offset + 4)
		}
		offset += 8
	}

	for (let i = 0; i < WEBGPU_MAX_POINT_LIGHTS; i++) {
		const light = input.pointLights[i]
		if (light) {
			data.set(
				[light.position[0], light.position[1], light.position[2], light.range],
				offset
			)
			data.set([light.color[0], light.color[1], light.color[2], 0], offset + 4)
		}
		offset += 8
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		const light = input.spotLights[i]
		if (light) {
			data.set(
				[light.position[0], light.position[1], light.position[2], light.range],
				offset
			)
			data.set(
				[light.direction[0], light.direction[1], light.direction[2], light.outerCos],
				offset + 4
			)
			data.set([light.color[0], light.color[1], light.color[2], light.innerCos], offset + 8)
		}
		offset += 12
	}

	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		const shadow = input.directionalShadows[i]
		if (shadow?.enabled && shadow.viewProjectionMatrix) {
			data.set(packMatrix4ForWGSL(shadow.viewProjectionMatrix), offset)
		}

		data.set(
			[
				shadow?.enabled ? 1 : 0,
				shadow?.depthBias ?? 0,
				shadow?.normalBias ?? 0,
				shadow?.normalBiasMin ?? 0,
			],
			offset + 16
		)
		data.set(
			[
				shadow?.pcfRadius ?? 0,
				shadow?.shadowStrength ?? 0,
				shadow?.shadowMapSize ?? 0,
				shadow?.atlasTileSize ?? 0,
			],
			offset + 20
		)
		offset += 24
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		const shadow = input.spotShadows[i]
		if (shadow?.enabled && shadow.viewProjectionMatrix) {
			data.set(packMatrix4ForWGSL(shadow.viewProjectionMatrix), offset)
		}

		data.set(
			[
				shadow?.enabled ? 1 : 0,
				shadow?.depthBias ?? 0,
				shadow?.normalBias ?? 0,
				shadow?.normalBiasMin ?? 0,
			],
			offset + 16
		)
		data.set(
			[
				shadow?.pcfRadius ?? 0,
				shadow?.shadowStrength ?? 0,
				shadow?.shadowMapSize ?? 0,
				shadow?.atlasTileSize ?? 0,
			],
			offset + 20
		)
		offset += 24
	}

	return data
}

export function remapClipSpaceDepth(clipZ: number, clipW: number): number {
	return clipZ * 0.5 + clipW * 0.5
}

export function packModelUniformData(
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	materialData: WebGPUMaterialUniformData
): Float32Array {
	const data = new Float32Array(WEBGPU_MODEL_UNIFORM_FLOATS)

	data.set(packMatrix4ForWGSL(modelMatrix), 0)
	data.set(packNormalMatrix4ForWGSL(normalMatrix), 16)
	data.set(materialData.baseColorFactor, 32)
	data.set(materialData.emissiveFactor, 36)
	data.set(materialData.surfaceParams0, 40)
	data.set(materialData.surfaceParams1, 44)
	data.set(materialData.surfaceParams2, 48)
	data.set(materialData.surfaceParams3, 52)
	data.set(materialData.specularColorFactor, 56)
	data.set(materialData.phongAmbientShininess, 60)
	data.set(materialData.phongSpecularShading, 64)
	data.set(materialData.sheenColorClearcoatNormalScale, 68)
	data.set(materialData.attenuationColor, 72)
	data.set(materialData.materialFlags, 76)

	let offset = 80
	for (const slot of materialData.textureSlots) {
		data.set(slot.transformA, offset)
		offset += 4
	}

	for (const slot of materialData.textureSlots) {
		data.set(slot.transformB, offset)
		offset += 4
	}

	return data
}

export function createTextureUploadData(texture: Texture): {
	data: Uint8Array
	bytesPerRow: number
	width: number
	height: number
} {
	const pixelData = toUint8TextureData(texture)
	const bytesPerPixel = 4
	const unalignedBytesPerRow = texture.width * bytesPerPixel
	const bytesPerRow = alignTo(unalignedBytesPerRow, 256)

	if (bytesPerRow === unalignedBytesPerRow) {
		return {
			data: pixelData,
			bytesPerRow,
			width: texture.width,
			height: texture.height,
		}
	}

	const padded = new Uint8Array(bytesPerRow * texture.height)
	for (let y = 0; y < texture.height; y++) {
		const srcStart = y * unalignedBytesPerRow
		const dstStart = y * bytesPerRow
		padded.set(pixelData.subarray(srcStart, srcStart + unalignedBytesPerRow), dstStart)
	}

	return {
		data: padded,
		bytesPerRow,
		width: texture.width,
		height: texture.height,
	}
}

export function alignTo(value: number, alignment: number): number {
	return Math.ceil(value / alignment) * alignment
}

function createMaterialTextureSlots(material: Material): WebGPUTextureSlotData[] {
	const mat = material as any
	const slots = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () =>
		createTextureSlot(null, 0, false)
	)

	slots[WEBGPU_TEXTURE_SLOT.BASE_COLOR] = createTextureSlot(
		material.map ?? null,
		mat.albedoMapUV ?? 0,
		false
	)
	slots[WEBGPU_TEXTURE_SLOT.METALLIC_ROUGHNESS] = createTextureSlot(
		mat.metallicRoughnessMap ?? null,
		mat.metallicRoughnessMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.NORMAL] = createTextureSlot(
		mat.normalMap ?? null,
		mat.normalMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.EMISSIVE] = createTextureSlot(
		mat.emissiveMap ?? null,
		mat.emissiveMapUV ?? 0,
		false
	)
	slots[WEBGPU_TEXTURE_SLOT.OCCLUSION] = createTextureSlot(
		mat.occlusionMap ?? null,
		mat.occlusionMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.SPECULAR] = createTextureSlot(
		mat.specularMap ?? null,
		mat.specularMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.SPECULAR_COLOR] = createTextureSlot(
		mat.specularColorMap ?? null,
		mat.specularColorMapUV ?? 0,
		false
	)
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT] = createTextureSlot(
		mat.clearcoatMap ?? null,
		mat.clearcoatMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT_ROUGHNESS] = createTextureSlot(
		mat.clearcoatRoughnessMap ?? null,
		mat.clearcoatRoughnessMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.CLEARCOAT_NORMAL] = createTextureSlot(
		mat.clearcoatNormalMap ?? null,
		mat.clearcoatNormalMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.SHEEN_COLOR] = createTextureSlot(
		mat.sheenColorMap ?? null,
		mat.sheenColorMapUV ?? 0,
		false
	)
	slots[WEBGPU_TEXTURE_SLOT.SHEEN_ROUGHNESS] = createTextureSlot(
		mat.sheenRoughnessMap ?? null,
		mat.sheenRoughnessMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.TRANSMISSION] = createTextureSlot(
		mat.transmissionMap ?? null,
		mat.transmissionMapUV ?? 0,
		true
	)
	slots[WEBGPU_TEXTURE_SLOT.THICKNESS] = createTextureSlot(
		mat.thicknessMap ?? null,
		mat.thicknessMapUV ?? 0,
		true
	)

	return slots
}

function createDisabledShadowData(): WebGPUShadowData {
	return {
		enabled: false,
		viewProjectionMatrix: null,
		depthBias: 0,
		normalBias: 0,
		normalBiasMin: 0,
		pcfRadius: 0,
		shadowStrength: 0,
		shadowMapSize: 0,
		atlasTileSize: 0,
		shadowMap: null,
	}
}

function resolveWebGPUShadowData(
	enableShadows: boolean,
	shadowMap?: ShadowMap
): WebGPUShadowData {
	if (!enableShadows || !shadowMap?.viewProjectionMatrix) {
		return createDisabledShadowData()
	}

	const size = Math.max(1, shadowMap.size | 0)
	const texelBias = (shadowMap.params.shadowTexelBias ?? 1.0) * (2.0 / size)
	const maxBias = shadowMap.params.shadowMaxBias ?? 0.05
	const depthBias =
		Math.min(
			maxBias,
			(shadowMap.params.shadowBias ?? 0.008) + texelBias
		) * 0.5
	const pcfRadius =
		shadowMap.params.shadowRadius && shadowMap.params.shadowRadius > 0
			? shadowMap.params.shadowRadius
			: Math.max(1, shadowMap.params.shadowPCF ?? 1)

	return {
		enabled: true,
		viewProjectionMatrix: shadowMap.viewProjectionMatrix,
		depthBias,
		normalBias: Math.max(0, shadowMap.params.shadowNormalBias ?? 1.0),
		normalBiasMin: Math.max(
			0,
			shadowMap.params.shadowNormalBiasMin ?? 0.05
		),
		pcfRadius: Math.max(1, pcfRadius),
		shadowStrength: clamp(shadowMap.params.shadowStrength ?? 1.0, 0, 1),
		shadowMapSize: size,
		atlasTileSize: size,
		shadowMap,
	}
}

function createTextureSlot(
	map: Texture | null | undefined,
	uvSet: number,
	fallbackLinear: boolean
): WebGPUTextureSlotData {
	if (!map) {
		return {
			map: null,
			transformA: [0, 0, 1, 1],
			transformB: [0, uvSet === 1 ? 1 : 0, fallbackLinear ? 1 : 0, 0],
		}
	}

	return {
		map,
		transformA: [map.offset.x, map.offset.y, map.repeat.x, map.repeat.y],
		transformB: [
			map.rotation,
			uvSet === 1 ? 1 : 0,
			map.colorSpace === 'sRGB' ? 0 : 1,
			0,
		],
	}
}

function getMaterialBaseColor(
	material: Material,
	isPBR: boolean
): [number, number, number] {
	if (isPBR) {
		const albedo = (material as any).albedo ?? { r: 255, g: 255, b: 255 }
		return [
			clamp(albedo.r / 255, 0, 1),
			clamp(albedo.g / 255, 0, 1),
			clamp(albedo.b / 255, 0, 1),
		]
	}

	const diffuse = (material as any).diffuse ?? { r: 255, g: 255, b: 255 }
	return [
		sRGBToLinear(clamp(diffuse.r / 255, 0, 1)),
		sRGBToLinear(clamp(diffuse.g / 255, 0, 1)),
		sRGBToLinear(clamp(diffuse.b / 255, 0, 1)),
	]
}

function getMaterialEmissive(
	material: Material,
	isPBR: boolean
): [number, number, number] {
	const emissive = (material as any).emissive
	if (!emissive) {
		return [0, 0, 0]
	}

	if (isPBR) {
		return [
			clamp(emissive.r / 255, 0, 1),
			clamp(emissive.g / 255, 0, 1),
			clamp(emissive.b / 255, 0, 1),
		]
	}

	return [
		sRGBToLinear(clamp(emissive.r / 255, 0, 1)),
		sRGBToLinear(clamp(emissive.g / 255, 0, 1)),
		sRGBToLinear(clamp(emissive.b / 255, 0, 1)),
	]
}

function getPhongLinearColor(color: {
	r: number
	g: number
	b: number
}): [number, number, number] {
	return [
		sRGBToLinear(clamp(color.r / 255, 0, 1)),
		sRGBToLinear(clamp(color.g / 255, 0, 1)),
		sRGBToLinear(clamp(color.b / 255, 0, 1)),
	]
}

function getPBRLinearColor(color: {
	r: number
	g: number
	b: number
}): [number, number, number] {
	return [
		clamp(color.r / 255, 0, 1),
		clamp(color.g / 255, 0, 1),
		clamp(color.b / 255, 0, 1),
	]
}

function resolveShadingMode(material: Material): number {
	switch (material.shading) {
		case 'PBR':
			return 1
		case 'Unlit':
			return 2
		case 'Flat':
			return 3
		default:
			return 0
	}
}

function pushMaterialWarnings(
	material: Material,
	warnings: WebGPUWarning[]
): void {
	const warn = (feature: string, message: string, enabled: boolean) => {
		if (!enabled) return
		warnings.push({
			key: `webgpu-material-${feature}:${material.type}:${material.name}`,
			message,
		})
	}

	warn(
		'reflectivity',
		`WebGPU backend does not support planar reflections yet; ignoring reflectivity on material ${material.name}`,
		(material.reflectivity ?? 0) > 0
	)
	warn(
		'mirror-plane',
		`WebGPU backend does not support mirrorPlane yet; rendering material ${material.name} as a regular lit surface`,
		!!material.mirrorPlane
	)
}

function toUint8TextureData(texture: Texture): Uint8Array {
	const data = texture.data

	if (!data) {
		return new Uint8Array(texture.width * texture.height * 4)
	}

	if (data instanceof Uint8Array && !(data instanceof Uint8ClampedArray)) {
		return data
	}

	if (data instanceof Uint8ClampedArray) {
		return new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
	}

	const converted = new Uint8Array(texture.width * texture.height * 4)
	for (let i = 0; i < converted.length; i++) {
		converted[i] = clamp(Math.round(data[i] * 255), 0, 255)
	}

	return converted
}

function normalizeVec3(value: IVector3): IVector3 {
	const length = Math.hypot(value.x, value.y, value.z) || 1
	return {
		x: value.x / length,
		y: value.y / length,
		z: value.z / length,
	}
}

function toLinearLightColor(
	color: { r: number; g: number; b: number },
	intensity: number
): [number, number, number] {
	return [
		sRGBToLinear(color.r / 255) * intensity,
		sRGBToLinear(color.g / 255) * intensity,
		sRGBToLinear(color.b / 255) * intensity,
	]
}

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value))
}
