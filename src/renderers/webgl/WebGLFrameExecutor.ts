import { CameraType } from '../../cameras/Camera'
import { ShaderMaterial } from '../../materials/ShaderMaterial'
import { AlphaMode, ShadingModel, type Material } from '../../materials/Material'
import { clamp, sRGBToLinear } from '../../maths/Common'
import type { Matrix4 } from '../../maths/Matrix4'
import type { Matrix3Arr } from '../../maths/types'
import type { DrawPacket, FrameContext, FramePass } from '../../pipeline/types'
import { collectWebGLLights, type WebGLLightState } from './WebGLLightCollector'
import { WebGLGeometryRegistry } from './WebGLGeometryRegistry'
import { WebGLProgramLibrary, type WebGLSceneProgram } from './WebGLProgramLibrary'
import { WebGLTextureRegistry } from './WebGLTextureRegistry'

type WarnFn = (key: string, message: string) => void

interface MaterialUniformState {
	shadingModel: number
	baseColor: [number, number, number, number]
	emissive: [number, number, number]
	pbr: [number, number, number, number]
	phong: [number, number, number, number]
	alpha: [number, number, number, number]
	baseMap: any | null
}

const SUPPORTED_STAGES = new Set<FramePass['stage']>([
	'main-opaque',
	'main-transparent',
	'gamma',
])

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext
	private _warn: WarnFn
	private _programs: WebGLProgramLibrary
	private _geometry: WebGLGeometryRegistry
	private _textures: WebGLTextureRegistry
	private _sceneFramebuffer: WebGLFramebuffer | null = null
	private _sceneColorTexture: WebGLTexture | null = null
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null
	private _fullscreenVao: WebGLVertexArrayObject | null = null
	private _width = 1
	private _height = 1
	private _presentedInFrame = false
	private _activeContext: FrameContext | null = null
	private _lightState: WebGLLightState | null = null

	constructor(gl: WebGL2RenderingContext, warn: WarnFn) {
		this._gl = gl
		this._warn = warn
		this._programs = new WebGLProgramLibrary(gl, warn)
		this._geometry = new WebGLGeometryRegistry(gl, warn)
		this._textures = new WebGLTextureRegistry(gl, warn)
		this._fullscreenVao = gl.createVertexArray()
	}

	public beginFrame(context: FrameContext): void {
		this._activeContext = context
		this._presentedInFrame = false
		this._width = Math.max(1, context.attachments.width | 0)
		this._height = Math.max(1, context.attachments.height | 0)
		this._ensureFrameTargets(this._width, this._height)
		this._lightState = collectWebGLLights(
			context.scene.lights,
			context.features.enableLighting,
			this._warn
		)

		const gl = this._gl
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer)
		gl.viewport(0, 0, this._width, this._height)
		gl.disable(gl.BLEND)
		gl.enable(gl.DEPTH_TEST)
		gl.depthMask(true)
		gl.clearColor(0, 0, 0, 1)
		gl.clearDepth(1)
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

		if (context.features.enableSkybox && context.scene.skybox) {
			this._renderSkybox(context)
		}
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		if (!SUPPORTED_STAGES.has(pass.stage)) {
			this._warn(
				`webgl-stage-unsupported-${pass.stage}`,
				`WebGL v1 does not support pass "${pass.stage}" yet; skipping`
			)
			return
		}

		switch (pass.stage) {
			case 'main-opaque':
				this._renderPackets(context, context.scene.opaquePackets, false)
				break
			case 'main-transparent':
				this._renderPackets(context, context.scene.transparentPackets, true)
				break
			case 'gamma':
				this._present(context.features.enableGamma !== false)
				break
		}
	}

	public endFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext?.features.enableGamma !== false)
		}
		this._activeContext = null
	}

	public resize(width: number, height: number): void {
		this._width = Math.max(1, width | 0)
		this._height = Math.max(1, height | 0)
		this._destroyFrameTargets()
	}

	public destroy(): void {
		this._destroyFrameTargets()
		if (this._fullscreenVao) {
			this._gl.deleteVertexArray(this._fullscreenVao)
			this._fullscreenVao = null
		}
		this._geometry.destroy()
		this._textures.destroy()
		this._programs.destroy()
		this._activeContext = null
	}

	private _renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): void {
		if (packets.length === 0) return
		if (!this._sceneFramebuffer) return

		const gl = this._gl
		const sceneProgram = this._programs.getSceneProgram()
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer)
		gl.useProgram(sceneProgram.program)
		gl.bindVertexArray(this._fullscreenVao)
		gl.activeTexture(gl.TEXTURE0)

		this._bindGlobalUniforms(sceneProgram, context)

		gl.enable(gl.DEPTH_TEST)
		gl.depthMask(!transparent)
		if (transparent) {
			gl.enable(gl.BLEND)
			gl.blendFuncSeparate(
				gl.SRC_ALPHA,
				gl.ONE_MINUS_SRC_ALPHA,
				gl.ONE,
				gl.ONE_MINUS_SRC_ALPHA
			)
		} else {
			gl.disable(gl.BLEND)
		}

		for (const packet of packets) {
			this._drawPacket(sceneProgram, context, packet, transparent)
		}

		gl.depthMask(true)
		gl.disable(gl.BLEND)
		gl.bindVertexArray(null)
	}

	private _drawPacket(
		sceneProgram: WebGLSceneProgram,
		context: FrameContext,
		packet: DrawPacket,
		transparentPass: boolean
	): void {
		const gl = this._gl
		const material = packet.material
		const alphaMode = material.alphaMode ?? AlphaMode.Opaque
		const requiresTransparent = alphaMode === AlphaMode.Blend
		if (transparentPass !== requiresTransparent) {
			return
		}

		if (packet.meshInstance.skeleton) {
			this._warn(
				`webgl-skinning-unsupported-${packet.meshInstance.id}`,
				`WebGL v1 does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
			)
			return
		}
		if (!isFiniteMatrix(packet.worldMatrix)) {
			this._warn(
				`webgl-world-matrix-invalid-${packet.id}`,
				`WebGL packet ${packet.id} has non-finite world matrix; skipping`
			)
			return
		}

		const geometry = this._geometry.getGeometry(packet)
		if (!geometry) {
			return
		}

		const uniforms = resolveMaterialUniforms(material, this._warn)
		const normalMatrix = toColumnMajorMat3(packet.normalMatrix)
		if (!normalMatrix) {
			this._warn(
				`webgl-normal-matrix-invalid-${packet.id}`,
				`WebGL packet ${packet.id} has invalid normal matrix; skipping`
			)
			return
		}

		const resolvedMap = this._textures.getBaseColorTexture(uniforms.baseMap)
		gl.activeTexture(gl.TEXTURE0)
		gl.bindTexture(gl.TEXTURE_2D, resolvedMap.texture)

		this._setCullMode(material)
		gl.bindVertexArray(geometry.vao)
		if (sceneProgram.uniforms.model) {
			gl.uniformMatrix4fv(
				sceneProgram.uniforms.model,
				false,
				toColumnMajorMat4(packet.worldMatrix)
			)
		}
		if (sceneProgram.uniforms.normalMatrix) {
			gl.uniformMatrix3fv(sceneProgram.uniforms.normalMatrix, false, normalMatrix)
		}
		if (sceneProgram.uniforms.shadingModel) {
			gl.uniform1i(sceneProgram.uniforms.shadingModel, uniforms.shadingModel)
		}
		if (sceneProgram.uniforms.baseColor) {
			gl.uniform4fv(sceneProgram.uniforms.baseColor, uniforms.baseColor)
		}
		if (sceneProgram.uniforms.emissive) {
			gl.uniform4f(
				sceneProgram.uniforms.emissive,
				uniforms.emissive[0],
				uniforms.emissive[1],
				uniforms.emissive[2],
				1
			)
		}
		if (sceneProgram.uniforms.pbr) {
			gl.uniform4fv(sceneProgram.uniforms.pbr, uniforms.pbr)
		}
		if (sceneProgram.uniforms.phong) {
			gl.uniform4fv(sceneProgram.uniforms.phong, uniforms.phong)
		}
		if (sceneProgram.uniforms.alpha) {
			gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha)
		}
		if (sceneProgram.uniforms.baseMap) {
			gl.uniform1i(sceneProgram.uniforms.baseMap, 0)
		}
		if (sceneProgram.uniforms.hasBaseMap) {
			gl.uniform1i(sceneProgram.uniforms.hasBaseMap, uniforms.baseMap ? 1 : 0)
		}
		if (sceneProgram.uniforms.baseMapIsLinear) {
			gl.uniform1i(
				sceneProgram.uniforms.baseMapIsLinear,
				resolvedMap.isLinear ? 1 : 0
			)
		}

		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		)
		gl.bindVertexArray(null)
	}

	private _bindGlobalUniforms(
		sceneProgram: WebGLSceneProgram,
		context: FrameContext
	): void {
		const gl = this._gl
		const uniforms = sceneProgram.uniforms
		const lights = this._lightState ?? {
			ambientColor: [0, 0, 0] as [number, number, number],
			directionalLights: [],
			pointLights: [],
			spotLights: [],
		}

		if (uniforms.viewProjection) {
			gl.uniformMatrix4fv(
				uniforms.viewProjection,
				false,
				toColumnMajorMat4(context.camera.viewProjectionMatrix)
			)
		}
		if (uniforms.cameraPosition) {
			const cameraPosition = context.camera.getWorldPosition()
			gl.uniform3f(
				uniforms.cameraPosition,
				cameraPosition.x,
				cameraPosition.y,
				cameraPosition.z
			)
		}
		if (uniforms.ambientColor) {
			gl.uniform3f(
				uniforms.ambientColor,
				lights.ambientColor[0],
				lights.ambientColor[1],
				lights.ambientColor[2]
			)
		}
		if (uniforms.enableLighting) {
			gl.uniform1i(uniforms.enableLighting, context.features.enableLighting ? 1 : 0)
		}

		if (uniforms.dirLightCount) {
			gl.uniform1i(uniforms.dirLightCount, lights.directionalLights.length)
		}
		if (uniforms.dirLightDirection) {
			gl.uniform4fv(
				uniforms.dirLightDirection,
				flattenVec4(lights.directionalLights, (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					0,
				])
			)
		}
		if (uniforms.dirLightColor) {
			gl.uniform4fv(
				uniforms.dirLightColor,
				flattenVec4(lights.directionalLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				])
			)
		}

		if (uniforms.pointLightCount) {
			gl.uniform1i(uniforms.pointLightCount, lights.pointLights.length)
		}
		if (uniforms.pointLightPositionRange) {
			gl.uniform4fv(
				uniforms.pointLightPositionRange,
				flattenVec4(lights.pointLights, (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				])
			)
		}
		if (uniforms.pointLightColor) {
			gl.uniform4fv(
				uniforms.pointLightColor,
				flattenVec4(lights.pointLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				])
			)
		}

		if (uniforms.spotLightCount) {
			gl.uniform1i(uniforms.spotLightCount, lights.spotLights.length)
		}
		if (uniforms.spotLightPositionRange) {
			gl.uniform4fv(
				uniforms.spotLightPositionRange,
				flattenVec4(lights.spotLights, (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				])
			)
		}
		if (uniforms.spotLightDirectionOuter) {
			gl.uniform4fv(
				uniforms.spotLightDirectionOuter,
				flattenVec4(lights.spotLights, (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					light.outerCos,
				])
			)
		}
		if (uniforms.spotLightColorInner) {
			gl.uniform4fv(
				uniforms.spotLightColorInner,
				flattenVec4(lights.spotLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					light.innerCos,
				])
			)
		}
	}

	private _renderSkybox(context: FrameContext): void {
		const skyboxTexture = context.scene.skybox
		if (!skyboxTexture || !this._fullscreenVao) return

		const gl = this._gl
		const skyboxProgram = this._programs.getSkyboxProgram()
		const resolved = this._textures.getSkyboxTexture(skyboxTexture)
		const view = context.camera.viewMatrix.elements
		const isOrthographic = context.camera.type === CameraType.Orthographic
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360)
		const aspect = context.camera.aspectRatio || this._width / this._height

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer)
		gl.useProgram(skyboxProgram.program)
		gl.bindVertexArray(this._fullscreenVao)
		gl.disable(gl.CULL_FACE)
		gl.disable(gl.BLEND)
		gl.disable(gl.DEPTH_TEST)
		gl.depthMask(false)

		gl.activeTexture(gl.TEXTURE0)
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture)
		if (skyboxProgram.uniforms.skyboxMap) {
			gl.uniform1i(skyboxProgram.uniforms.skyboxMap, 0)
		}
		if (skyboxProgram.uniforms.skyboxBasisRight) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisRight,
				view[0][0],
				view[0][1],
				view[0][2],
				tanHalfFov
			)
		}
		if (skyboxProgram.uniforms.skyboxBasisUp) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisUp,
				view[1][0],
				view[1][1],
				view[1][2],
				aspect
			)
		}
		if (skyboxProgram.uniforms.skyboxBasisBackward) {
			gl.uniform3f(
				skyboxProgram.uniforms.skyboxBasisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			)
		}
		if (skyboxProgram.uniforms.skyboxIsOrthographic) {
			gl.uniform1f(
				skyboxProgram.uniforms.skyboxIsOrthographic,
				isOrthographic ? 1 : 0
			)
		}
		if (skyboxProgram.uniforms.skyboxMapIsLinear) {
			gl.uniform1i(skyboxProgram.uniforms.skyboxMapIsLinear, resolved.isLinear ? 1 : 0)
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3)

		gl.depthMask(true)
		gl.enable(gl.DEPTH_TEST)
		gl.bindVertexArray(null)
	}

	private _present(applyGamma: boolean): void {
		if (!this._sceneColorTexture || !this._fullscreenVao) return
		const gl = this._gl
		const presentProgram = this._programs.getPresentProgram()
		gl.bindFramebuffer(gl.FRAMEBUFFER, null)
		gl.viewport(0, 0, this._width, this._height)
		gl.useProgram(presentProgram.program)
		gl.bindVertexArray(this._fullscreenVao)
		gl.disable(gl.CULL_FACE)
		gl.disable(gl.DEPTH_TEST)
		gl.disable(gl.BLEND)
		gl.activeTexture(gl.TEXTURE0)
		gl.bindTexture(gl.TEXTURE_2D, this._sceneColorTexture)
		if (presentProgram.uniforms.sourceMap) {
			gl.uniform1i(presentProgram.uniforms.sourceMap, 0)
		}
		if (presentProgram.uniforms.applyGamma) {
			gl.uniform1i(presentProgram.uniforms.applyGamma, applyGamma ? 1 : 0)
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3)
		gl.bindVertexArray(null)
		this._presentedInFrame = true
	}

	private _ensureFrameTargets(width: number, height: number): void {
		if (
			this._sceneFramebuffer &&
			this._sceneColorTexture &&
			this._sceneDepthBuffer &&
			this._width === width &&
			this._height === height
		) {
			return
		}
		this._destroyFrameTargets()
		const gl = this._gl

		const framebuffer = gl.createFramebuffer()
		const colorTexture = gl.createTexture()
		const depthBuffer = gl.createRenderbuffer()
		if (!framebuffer || !colorTexture || !depthBuffer) {
			if (framebuffer) gl.deleteFramebuffer(framebuffer)
			if (colorTexture) gl.deleteTexture(colorTexture)
			if (depthBuffer) gl.deleteRenderbuffer(depthBuffer)
			throw new Error('Failed to create WebGL frame targets')
		}

		gl.bindTexture(gl.TEXTURE_2D, colorTexture)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA,
			width,
			height,
			0,
			gl.RGBA,
			gl.UNSIGNED_BYTE,
			null
		)
		gl.bindTexture(gl.TEXTURE_2D, null)

		gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer)
		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height)
		gl.bindRenderbuffer(gl.RENDERBUFFER, null)

		gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			colorTexture,
			0
		)
		gl.framebufferRenderbuffer(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.RENDERBUFFER,
			depthBuffer
		)
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER)
		gl.bindFramebuffer(gl.FRAMEBUFFER, null)
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.deleteFramebuffer(framebuffer)
			gl.deleteTexture(colorTexture)
			gl.deleteRenderbuffer(depthBuffer)
			throw new Error(
				`WebGL framebuffer is incomplete (status=0x${status.toString(16)})`
			)
		}

		this._sceneFramebuffer = framebuffer
		this._sceneColorTexture = colorTexture
		this._sceneDepthBuffer = depthBuffer
	}

	private _destroyFrameTargets(): void {
		const gl = this._gl
		if (this._sceneFramebuffer) {
			gl.deleteFramebuffer(this._sceneFramebuffer)
			this._sceneFramebuffer = null
		}
		if (this._sceneColorTexture) {
			gl.deleteTexture(this._sceneColorTexture)
			this._sceneColorTexture = null
		}
		if (this._sceneDepthBuffer) {
			gl.deleteRenderbuffer(this._sceneDepthBuffer)
			this._sceneDepthBuffer = null
		}
	}

	private _setCullMode(material: Material): void {
		const gl = this._gl
		if (material.doubleSided || material.cullMode === 'none') {
			gl.disable(gl.CULL_FACE)
			return
		}
		gl.enable(gl.CULL_FACE)
		gl.frontFace(gl.CCW)
		if (material.cullMode === 'front') {
			gl.cullFace(gl.FRONT)
		} else {
			gl.cullFace(gl.BACK)
		}
	}
}

function resolveMaterialUniforms(
	material: Material,
	warn: WarnFn
): MaterialUniformState {
	const isShaderMaterial = material instanceof ShaderMaterial
	if (isShaderMaterial) {
		warn(
			`webgl-shader-material-fallback-${material.name}`,
			`WebGL v1 does not support ShaderMaterial custom shader path yet; falling back to built-in shading for ${material.name}`
		)
	}
	const isPBR = material.shading === ShadingModel.PBR || material.type === 'PBR'
	const isUnlit = material.shading === ShadingModel.Unlit

	let baseColor: [number, number, number] = [1, 1, 1]
	let emissive: [number, number, number] = [0, 0, 0]
	let roughness = 0.5
	let metalness = 0
	let reflectance = 0.5
	let shininess = 32
	let baseMap: any | null = material.map ?? null

	if (isPBR) {
		const pbr = material as any
		const albedo = pbr.albedo ?? { r: 255, g: 255, b: 255 }
		baseColor = [
			clamp((albedo.r ?? 255) / 255, 0, 1),
			clamp((albedo.g ?? 255) / 255, 0, 1),
			clamp((albedo.b ?? 255) / 255, 0, 1),
		]
		const emissiveColor = pbr.emissive ?? { r: 0, g: 0, b: 0 }
		const emissiveIntensity = clamp(pbr.emissiveIntensity ?? 1, 0, 64)
		emissive = [
			clamp((emissiveColor.r ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.g ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.b ?? 0) / 255, 0, 1) * emissiveIntensity,
		]
		roughness = clamp(pbr.roughness ?? 0.5, 0.04, 1)
		metalness = clamp(pbr.metalness ?? 0, 0, 1)
		reflectance = clamp(pbr.reflectance ?? 0.5, 0, 1)
		baseMap = pbr.map ?? baseMap
	} else {
		const basic = material as any
		const diffuse = basic.diffuse ?? { r: 255, g: 255, b: 255 }
		baseColor = [
			sRGBToLinear(clamp((diffuse.r ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.g ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.b ?? 255) / 255, 0, 1)),
		]
		const emissiveColor = basic.emissive
		if (emissiveColor) {
			const emissiveIntensity = clamp(basic.emissiveIntensity ?? 1, 0, 64)
			emissive = [
				sRGBToLinear(clamp((emissiveColor.r ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.g ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.b ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
			]
		}
		shininess = Math.max(1, basic.shininess ?? 32)
	}

	const opacity = clamp(material.opacity ?? 1, 0, 1)
	const alphaCutoff = clamp(material.alphaCutoff ?? 0.5, 0, 1)
	const alphaModeMask = material.alphaMode === AlphaMode.Mask ? 1 : 0

	return {
		shadingModel: isUnlit ? 2 : isPBR ? 1 : 0,
		baseColor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissive,
		pbr: [roughness, metalness, reflectance, 0],
		phong: [shininess, 0, 0, 0],
		alpha: [alphaCutoff, alphaModeMask, 0, 0],
		baseMap,
	}
}

function flattenVec4<T>(
	values: T[],
	mapper: (value: T) => [number, number, number, number]
): Float32Array {
	const packed = new Float32Array(16)
	const count = Math.min(4, values.length)
	for (let i = 0; i < count; i++) {
		const value = mapper(values[i])
		const offset = i * 4
		packed[offset] = value[0]
		packed[offset + 1] = value[1]
		packed[offset + 2] = value[2]
		packed[offset + 3] = value[3]
	}
	return packed
}

function toColumnMajorMat4(matrix: Matrix4 | number[][]): Float32Array {
	const elements = matrix instanceof Array ? matrix : matrix.elements
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

function toColumnMajorMat3(matrix: Matrix4 | Matrix3Arr): Float32Array | null {
	const rows: number[][] =
		matrix instanceof Array ? matrix : (matrix as Matrix4).elements
	if (!rows || rows.length < 3) return null
	const values = [
		rows[0][0],
		rows[1][0],
		rows[2][0],
		rows[0][1],
		rows[1][1],
		rows[2][1],
		rows[0][2],
		rows[1][2],
		rows[2][2],
	]
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null
		}
	}
	return new Float32Array(values)
}

function isFiniteMatrix(matrix: Matrix4): boolean {
	const elements = matrix.elements
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			if (!Number.isFinite(elements[row][col])) {
				return false
			}
		}
	}
	return true
}
