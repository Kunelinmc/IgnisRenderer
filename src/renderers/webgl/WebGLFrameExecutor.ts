import { CameraType } from "../../cameras/Camera";
import { isShadowCastingLight } from "../../lights";
import { ParticleBlendMode } from "../../particles";
import {
	AlphaMode,
	ShadingModel,
	type Material,
} from "../../materials/Material";
import { clamp, sRGBToLinear } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import {
	resolveShadowCasterBounds,
	syncShadowMapRegistry,
	updateShadowMapMetadata,
} from "../../pipeline/ShadowMetadata";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type FrameContext,
	type FramePass,
	type ParticleRenderBatch,
	type TAAOptions,
} from "../../pipeline/types";
import {
	collectWebGLLights,
	type WebGLLightState,
	type WebGLShadowData,
} from "./WebGLLightCollector";
import { WebGLGeometryRegistry } from "./WebGLGeometryRegistry";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
	WEBGL_SHADOW_ATLAS_COLUMNS,
	WEBGL_SHADOW_ATLAS_ROWS,
} from "./constants";
import {
	WebGLProgramLibrary,
	type WebGLSceneProgram,
	type WebGLShadowDepthProgram,
} from "./WebGLProgramLibrary";
import { WebGLTextureRegistry } from "./WebGLTextureRegistry";

type WarnFn = (key: string, message: string) => void;

interface MaterialUniformState {
	shadingModel: number;
	baseColor: [number, number, number, number];
	emissive: [number, number, number];
	pbr: [number, number, number, number];
	phong: [number, number, number, number];
	alpha: [number, number, number, number];
	baseMap: any | null;
}

const SUPPORTED_STAGES = new Set<FramePass["stage"]>([
	"shadow",
	"main-opaque",
	"main-transparent",
	"particles",
	"fxaa",
	"taa",
	"gamma",
]);

const TAA_HALTON_SAMPLE_COUNT = 16;

function computeHaltonJitterNDC(index: number, width: number, height: number): [number, number] {
	const haltonX = [0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375, 0.03125];
	const haltonY = [0.333333, 0.666667, 0.111111, 0.444444, 0.777778, 0.222222, 0.555556, 0.888889, 0.037037, 0.37037, 0.703704, 0.148148, 0.481481, 0.814815, 0.259259, 0.592593];
	
	const idx = index % TAA_HALTON_SAMPLE_COUNT;
	return [
		((haltonX[idx] - 0.5) / width) * 2.0,
		((haltonY[idx] - 0.5) / height) * 2.0
	];
}

const PARTICLE_QUAD_VERTICES = new Float32Array([
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	-0.5,
	1,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	-0.5,
	0,
	1,
	0.5,
	0.5,
	1,
	0,
	-0.5,
	0.5,
	0,
	0,
]);

const PARTICLE_QUAD_STRIDE = 16;
const PARTICLE_INSTANCE_FLOATS = 13;
const PARTICLE_INSTANCE_STRIDE = PARTICLE_INSTANCE_FLOATS * 4;
const PARTICLE_INITIAL_CAPACITY = 256;
const PARTICLE_MAX_INSTANCES_PER_DRAW = 1 << 16;

export class WebGLFrameExecutor {
	private _gl: WebGL2RenderingContext;
	private _warn: WarnFn;
	private _programs: WebGLProgramLibrary;
	private _geometry: WebGLGeometryRegistry;
	private _textures: WebGLTextureRegistry;
	private _sceneFramebuffer: WebGLFramebuffer | null = null;
	private _sceneColorTexture: WebGLTexture | null = null;
	private _sceneMotionTexture: WebGLTexture | null = null;
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();
	private _taaHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaMotionHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaHistoryIndex = 0;
	private _taaJitter = new Float32Array(4); // currX, currY, prevX, prevY
	private _taaFrameIndex = 0;
	private _prevViewProjection: Float32Array | null = null;
	private _modelMatrixCache = new Map<string, Float32Array>();
	private _modelMatrixKeysThisFrame = new Set<string>();
	private _postFramebuffer: WebGLFramebuffer | null = null;
	private _postColorTexture: WebGLTexture | null = null;
	private _presentSourceTexture: WebGLTexture | null = null;
	private _fullscreenVao: WebGLVertexArrayObject | null = null;
	private _particleVao: WebGLVertexArrayObject | null = null;
	private _particleQuadBuffer: WebGLBuffer | null = null;
	private _particleInstanceBuffer: WebGLBuffer | null = null;
	private _particleInstanceCapacity = 0;
	private _particleScratch = new Float32Array(0);
	private _width = 1;
	private _height = 1;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _maxTextureSize: number;
	private _maxRenderbufferSize: number;
	private _presentedInFrame = false;
	private _activeContext: FrameContext | null = null;
	private _lightState: WebGLLightState | null = null;

	constructor(gl: WebGL2RenderingContext, warn: WarnFn) {
		this._gl = gl;
		this._warn = warn;
		this._programs = new WebGLProgramLibrary(gl, warn);
		this._geometry = new WebGLGeometryRegistry(gl, warn);
		this._textures = new WebGLTextureRegistry(gl, warn);
		this._fullscreenVao = gl.createVertexArray();
		this._maxTextureSize = this._resolveLimit(gl.MAX_TEXTURE_SIZE, 4096);
		this._maxRenderbufferSize = this._resolveLimit(
			gl.MAX_RENDERBUFFER_SIZE,
			4096
		);
	}

	public beginFrame(context: FrameContext): void {
		this._activeContext = context;
		this._presentedInFrame = false;
		this._modelMatrixKeysThisFrame.clear();
		this._width = toSafeDimension(context.attachments.width);
		this._height = toSafeDimension(context.attachments.height);
		this._ensureFrameTargets(this._width, this._height);
		this._presentSourceTexture = this._sceneColorTexture;
		this._syncShadowMetadata(context);
		this._lightState = collectWebGLLights(
			context.scene.lights,
			context.features.enableLighting,
			this._warn,
			context.features.enableShadows,
			context.shadowMaps
		);
		
		if (context.features.enableTAA) {
			this._taaJitter[2] = this._taaJitter[0];
			this._taaJitter[3] = this._taaJitter[1];
			const nextJitter = computeHaltonJitterNDC(this._taaFrameIndex++, this._width, this._height);
			this._taaJitter[0] = nextJitter[0];
			this._taaJitter[1] = nextJitter[1];
		} else {
			this._taaJitter.fill(0);
			this._taaFrameIndex = 0;
		}

		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.viewport(0, 0, this._width, this._height);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);
		gl.clearColor(0, 0, 0, 1);
		gl.clearDepth(1);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		if (context.features.enableSkybox && context.scene.skybox) {
			this._renderSkybox(context);
		}
	}

	public executePass(pass: FramePass, context: FrameContext): void {
		if (!SUPPORTED_STAGES.has(pass.stage)) {
			this._warn(
				`webgl-stage-unsupported-${pass.stage}`,
				`WebGL v1 does not support pass "${pass.stage}" yet; skipping`
			);
			return;
		}

		switch (pass.stage) {
			case "shadow":
				this._renderShadows(context);
				break;
			case "main-opaque":
				this._renderPackets(context, context.scene.opaquePackets, false);
				break;
			case "main-transparent":
				this._renderPackets(context, context.scene.transparentPackets, true);
				break;
			case "particles":
				this._renderParticles(context);
				break;
			case "fxaa":
				this._applyFXAA();
				break;
			case "taa":
				this._applyTAA(context.features.taaOptions);
				break;
			case "gamma":
				this._present(context.features.enableGamma !== false);
				break;
		}
	}

	public endFrame(): void {
		if (!this._presentedInFrame) {
			this._present(this._activeContext?.features.enableGamma !== false);
		}
		this._pruneModelMatrixCache();
		this._activeContext = null;
	}

	public resize(width: number, height: number): void {
		this._width = toSafeDimension(width);
		this._height = toSafeDimension(height);
		this._destroyFrameTargets();
	}

	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyShadowTargets();
		this._destroyParticleResources();
		this._modelMatrixCache.clear();
		this._modelMatrixKeysThisFrame.clear();
		if (this._fullscreenVao) {
			this._gl.deleteVertexArray(this._fullscreenVao);
			this._fullscreenVao = null;
		}
		this._geometry.destroy();
		this._textures.destroy();
		this._programs.destroy();
		this._activeContext = null;
	}

	private _syncShadowMetadata(context: FrameContext): void {
		const shadowLights = context.scene.lights.filter(isShadowCastingLight);
		syncShadowMapRegistry(context.shadowMaps, shadowLights);

		if (!context.features.enableShadows) {
			return;
		}

		const shadowCasterBounds = resolveShadowCasterBounds(
			context.scene.shadowCasterPackets,
			context.scene.sceneBounds,
			context.scene.camera
		);
		for (const light of shadowLights) {
			const shadowMap = context.shadowMaps.get(light);
			if (!shadowMap) continue;
			updateShadowMapMetadata(shadowMap, light, shadowCasterBounds);
		}
	}

	private _renderShadows(context: FrameContext): void {
		if (!context.features.enableShadows) {
			return;
		}
		const lights = this._lightState;
		if (!lights) {
			return;
		}
		const maxShadowSize = Math.max(
			getMaxShadowSize(lights.directionalShadows),
			getMaxShadowSize(lights.spotShadows)
		);
		if (maxShadowSize <= 0 || context.scene.shadowCasterPackets.length <= 0) {
			this._shadowAtlasTileSize = 0;
			return;
		}

		this._ensureShadowTargets(maxShadowSize);
		if (!this._shadowFramebuffer || !this._shadowAtlasTexture) {
			this._shadowAtlasTileSize = 0;
			return;
		}

		this._shadowAtlasTileSize = maxShadowSize;
		for (const shadow of lights.directionalShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}
		for (const shadow of lights.spotShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}

		const gl = this._gl;
		const shadowProgram = this._programs.getShadowDepthProgram();
		const packets = context.scene.shadowCasterPackets;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFramebuffer);
		gl.useProgram(shadowProgram.program);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);
		gl.colorMask(false, false, false, false);
		gl.enable(gl.SCISSOR_TEST);
		gl.clearDepth(1);
		gl.clear(gl.DEPTH_BUFFER_BIT);

		const directionalCount = Math.min(
			WEBGL_MAX_DIRECTIONAL_LIGHTS,
			lights.directionalShadows.length
		);
		for (let i = 0; i < directionalCount; i++) {
			this._renderShadowSlice(
				shadowProgram,
				packets,
				lights.directionalShadows[i],
				i,
				0
			);
		}

		const spotCount = Math.min(WEBGL_MAX_SPOT_LIGHTS, lights.spotShadows.length);
		for (let i = 0; i < spotCount; i++) {
			this._renderShadowSlice(
				shadowProgram,
				packets,
				lights.spotShadows[i],
				i,
				1
			);
		}

		gl.disable(gl.SCISSOR_TEST);
		gl.colorMask(true, true, true, true);
		gl.bindVertexArray(null);
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.viewport(0, 0, this._width, this._height);
	}

	private _renderShadowSlice(
		shadowProgram: WebGLShadowDepthProgram,
		packets: DrawPacket[],
		shadow: WebGLShadowData | undefined,
		tileX: number,
		tileY: number
	): void {
		if (!shadow?.enabled || !shadow.viewProjectionMatrix) {
			return;
		}

		const shadowSize = Math.max(1, shadow.shadowMapSize | 0);
		const viewportX = tileX * this._shadowAtlasTileSize;
		const viewportY = tileY * this._shadowAtlasTileSize;
		const gl = this._gl;
		gl.viewport(viewportX, viewportY, shadowSize, shadowSize);
		gl.scissor(viewportX, viewportY, shadowSize, shadowSize);
		gl.clear(gl.DEPTH_BUFFER_BIT);

		for (const packet of packets) {
			this._drawShadowPacket(shadowProgram, packet, shadow.viewProjectionMatrix);
		}
	}

	private _drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		if (packet.meshInstance.skeleton) {
			this._warn(
				"webgl-shadow-skinning-unsupported",
				`WebGL v1 shadow pass does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
			);
			return;
		}
		if (!isFiniteMatrix(packet.worldMatrix)) {
			return;
		}

		const geometry = this._geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._gl.TRIANGLES) {
			return;
		}

		Matrix4.multiply(
			viewProjectionMatrix,
			packet.worldMatrix,
			this._shadowMvpMatrix
		);
		const gl = this._gl;
		if (shadowProgram.uniforms.mvp) {
			gl.uniformMatrix4fv(
				shadowProgram.uniforms.mvp,
				false,
				toColumnMajorMat4(this._shadowMvpMatrix)
			);
		}

		this._setCullMode(packet.material);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		);
		gl.bindVertexArray(null);
	}

	private _ensureShadowTargets(tileSize: number): void {
		if (
			this._shadowFramebuffer &&
			this._shadowAtlasTexture &&
			this._shadowAtlasTileSize === tileSize
		) {
			return;
		}

		const atlasWidth = tileSize * WEBGL_SHADOW_ATLAS_COLUMNS;
		const atlasHeight = tileSize * WEBGL_SHADOW_ATLAS_ROWS;
		if (atlasWidth > this._maxTextureSize || atlasHeight > this._maxTextureSize) {
			throw new Error(
				`WebGL shadow atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._maxTextureSize}`
			);
		}

		this._destroyShadowTargets();
		const gl = this._gl;
		const shadowTexture = gl.createTexture();
		const shadowFramebuffer = gl.createFramebuffer();
		if (!shadowTexture || !shadowFramebuffer) {
			if (shadowTexture) gl.deleteTexture(shadowTexture);
			if (shadowFramebuffer) gl.deleteFramebuffer(shadowFramebuffer);
			throw new Error("Failed to create WebGL shadow atlas targets");
		}

		gl.bindTexture(gl.TEXTURE_2D, shadowTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.DEPTH_COMPONENT24,
			atlasWidth,
			atlasHeight,
			0,
			gl.DEPTH_COMPONENT,
			gl.UNSIGNED_INT,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.TEXTURE_2D,
			shadowTexture,
			0
		);
		gl.drawBuffers([gl.NONE]);
		gl.readBuffer(gl.NONE);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			throw new Error(
				`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowAtlasTileSize = tileSize;
	}

	private _destroyShadowTargets(): void {
		const gl = this._gl;
		if (this._shadowFramebuffer) {
			gl.deleteFramebuffer(this._shadowFramebuffer);
			this._shadowFramebuffer = null;
		}
		if (this._shadowAtlasTexture) {
			gl.deleteTexture(this._shadowAtlasTexture);
			this._shadowAtlasTexture = null;
		}
		this._shadowAtlasTileSize = 0;
	}

	private _renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean
	): void {
		if (packets.length === 0) return;
		if (!this._sceneFramebuffer) return;

		const gl = this._gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.activeTexture(gl.TEXTURE0);

		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(!transparent);
		if (transparent) {
			gl.enable(gl.BLEND);
			gl.blendFuncSeparate(
				gl.SRC_ALPHA,
				gl.ONE_MINUS_SRC_ALPHA,
				gl.ONE,
				gl.ONE_MINUS_SRC_ALPHA
			);
		} else {
			gl.disable(gl.BLEND);
		}

		let activeProgram: WebGLSceneProgram | null = null;
		for (const packet of packets) {
			const sceneProgram = this._programs.getSceneProgram(packet.material);
			if (activeProgram !== sceneProgram) {
				gl.useProgram(sceneProgram.program);
				this._bindGlobalUniforms(sceneProgram, context);
				activeProgram = sceneProgram;
			}
			this._drawPacket(sceneProgram, packet, transparent, context);
		}

		if (this._prevViewProjection) {
			this._prevViewProjection.set(toColumnMajorMat4(context.camera.viewProjectionMatrix));
		} else {
			this._prevViewProjection = toColumnMajorMat4(context.camera.viewProjectionMatrix);
		}

		gl.depthMask(true);
		gl.disable(gl.BLEND);
		gl.bindVertexArray(null);
	}

	private _drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext
	): void {
		const gl = this._gl;
		const material = packet.material;
		const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
		const requiresTransparent = alphaMode === AlphaMode.Blend;
		if (transparentPass !== requiresTransparent) {
			return;
		}

		if (packet.meshInstance.skeleton) {
			this._warn(
				"webgl-skinning-unsupported",
				`WebGL v1 does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
			);
			return;
		}
		if (!isFiniteMatrix(packet.worldMatrix)) {
			this._warn(
				"webgl-world-matrix-invalid",
				`WebGL packet ${packet.id} has non-finite world matrix; skipping`
			);
			return;
		}

		const geometry = this._geometry.getGeometry(packet);
		if (!geometry) {
			return;
		}

		const uniforms = resolveMaterialUniforms(material);
		const normalMatrix = toColumnMajorMat3(packet.normalMatrix);
		if (!normalMatrix) {
			this._warn(
				"webgl-normal-matrix-invalid",
				`WebGL packet ${packet.id} has invalid normal matrix; skipping`
			);
			return;
		}

		const resolvedMap = this._textures.getBaseColorTexture(uniforms.baseMap);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolvedMap.texture);

		this._setCullMode(material);
		gl.bindVertexArray(geometry.vao);
		if (sceneProgram.uniforms.model) {
			gl.uniformMatrix4fv(
				sceneProgram.uniforms.model,
				false,
				toColumnMajorMat4(packet.worldMatrix)
			);
		}
		if (sceneProgram.uniforms.normalMatrix) {
			gl.uniformMatrix3fv(
				sceneProgram.uniforms.normalMatrix,
				false,
				normalMatrix
			);
		}
		if (sceneProgram.uniforms.prevModel) {
			const cacheKey = packet.id;
			this._modelMatrixKeysThisFrame.add(cacheKey);
			let cached = this._modelMatrixCache.get(cacheKey);
			gl.uniformMatrix4fv(
				sceneProgram.uniforms.prevModel,
				false,
				cached ?? toColumnMajorMat4(packet.worldMatrix)
			);
			if (!cached) {
				cached = toColumnMajorMat4(packet.worldMatrix);
				this._modelMatrixCache.set(cacheKey, cached);
			} else {
				cached.set(toColumnMajorMat4(packet.worldMatrix));
			}
		}
		if (sceneProgram.uniforms.shadingModel) {
			gl.uniform1i(sceneProgram.uniforms.shadingModel, uniforms.shadingModel);
		}
		if (sceneProgram.uniforms.baseColor) {
			gl.uniform4fv(sceneProgram.uniforms.baseColor, uniforms.baseColor);
		}
		if (sceneProgram.uniforms.emissive) {
			gl.uniform4f(
				sceneProgram.uniforms.emissive,
				uniforms.emissive[0],
				uniforms.emissive[1],
				uniforms.emissive[2],
				1
			);
		}
		if (sceneProgram.uniforms.pbr) {
			gl.uniform4fv(sceneProgram.uniforms.pbr, uniforms.pbr);
		}
		if (sceneProgram.uniforms.phong) {
			gl.uniform4fv(sceneProgram.uniforms.phong, uniforms.phong);
		}
		if (sceneProgram.uniforms.alpha) {
			gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha);
		}
		if (sceneProgram.uniforms.baseMap) {
			gl.uniform1i(sceneProgram.uniforms.baseMap, 0);
		}
		if (sceneProgram.uniforms.hasBaseMap) {
			gl.uniform1i(sceneProgram.uniforms.hasBaseMap, uniforms.baseMap ? 1 : 0);
		}
		if (sceneProgram.uniforms.baseMapIsLinear) {
			gl.uniform1i(
				sceneProgram.uniforms.baseMapIsLinear,
				resolvedMap.isLinear ? 1 : 0
			);
		}
		if (sceneProgram.uniforms.doubleSided) {
			gl.uniform1i(
				sceneProgram.uniforms.doubleSided,
				material.doubleSided || material.cullMode === "none" ? 1 : 0
			);
		}

		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		);
		gl.bindVertexArray(null);
	}

	private _renderParticles(context: FrameContext): void {
		if (!this._sceneFramebuffer) return;

		const batches = context.transient.get(
			PARTICLE_TRANSIENT_BATCHES_KEY
		) as ParticleRenderBatch[] | undefined;
		if (!Array.isArray(batches) || batches.length === 0) {
			return;
		}

		this._ensureParticleResources();
		if (
			!this._particleVao ||
			!this._particleQuadBuffer ||
			!this._particleInstanceBuffer
		) {
			return;
		}

		const gl = this._gl;
		const particleProgram = this._programs.getParticleProgram();
		const view = context.camera.viewMatrix.elements;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.useProgram(particleProgram.program);
		gl.bindVertexArray(this._particleVao);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(false);
		gl.disable(gl.CULL_FACE);
		gl.enable(gl.BLEND);

		if (particleProgram.uniforms.viewProjection) {
			gl.uniformMatrix4fv(
				particleProgram.uniforms.viewProjection,
				false,
				toColumnMajorMat4(context.camera.viewProjectionMatrix)
			);
		}
		if (particleProgram.uniforms.basisRight) {
			gl.uniform3f(
				particleProgram.uniforms.basisRight,
				view[0][0],
				view[0][1],
				view[0][2]
			);
		}
		if (particleProgram.uniforms.basisUp) {
			gl.uniform3f(
				particleProgram.uniforms.basisUp,
				view[1][0],
				view[1][1],
				view[1][2]
			);
		}
		if (particleProgram.uniforms.particleMap) {
			gl.uniform1i(particleProgram.uniforms.particleMap, 0);
		}

		for (const batch of batches) {
			const preflightCount = Math.min(
				PARTICLE_MAX_INSTANCES_PER_DRAW,
				batch?.particles?.length ?? 0
			);
			if (preflightCount <= 0) {
				continue;
			}

			this._ensureParticleCapacity(preflightCount);
			const instanceCount = this._writeParticleInstances(batch);
			if (instanceCount <= 0 || !this._particleInstanceBuffer) {
				continue;
			}
			gl.bindBuffer(gl.ARRAY_BUFFER, this._particleInstanceBuffer);
			gl.bufferSubData(
				gl.ARRAY_BUFFER,
				0,
				this._particleScratch.subarray(0, instanceCount * PARTICLE_INSTANCE_FLOATS)
			);

			const resolvedTexture = this._textures.getBaseColorTexture(
				batch.texture ?? null
			);
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, resolvedTexture.texture);
			if (particleProgram.uniforms.mapIsLinear) {
				gl.uniform1i(
					particleProgram.uniforms.mapIsLinear,
					resolvedTexture.isLinear ? 1 : 0
				);
			}

			const uvTransform = resolveTextureUVTransform(batch.texture);
			if (particleProgram.uniforms.uvTransformA) {
				gl.uniform4f(
					particleProgram.uniforms.uvTransformA,
					uvTransform.repeatX,
					uvTransform.repeatY,
					uvTransform.offsetX,
					uvTransform.offsetY
				);
			}
			if (particleProgram.uniforms.uvTransformB) {
				gl.uniform2f(
					particleProgram.uniforms.uvTransformB,
					uvTransform.cosRotation,
					uvTransform.sinRotation
				);
			}

			if (batch.blendMode === ParticleBlendMode.Additive) {
				gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
			} else {
				gl.blendFuncSeparate(
					gl.SRC_ALPHA,
					gl.ONE_MINUS_SRC_ALPHA,
					gl.ONE,
					gl.ONE_MINUS_SRC_ALPHA
				);
			}

			gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, instanceCount);
		}

		gl.blendFuncSeparate(
			gl.SRC_ALPHA,
			gl.ONE_MINUS_SRC_ALPHA,
			gl.ONE,
			gl.ONE_MINUS_SRC_ALPHA
		);
		gl.depthMask(true);
		gl.disable(gl.BLEND);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		gl.bindVertexArray(null);
	}

	private _writeParticleInstances(batch: ParticleRenderBatch): number {
		const particles = batch.particles;
		if (!Array.isArray(particles) || particles.length === 0) {
			return 0;
		}

		let cappedCount = particles.length;
		if (cappedCount > PARTICLE_MAX_INSTANCES_PER_DRAW) {
			this._warn(
				"webgl-particle-cap",
				`WebGL particle pass truncates system "${batch.systemId}" to ${PARTICLE_MAX_INSTANCES_PER_DRAW} instances per draw`
			);
			cappedCount = PARTICLE_MAX_INSTANCES_PER_DRAW;
		}

		if (this._particleScratch.length < cappedCount * PARTICLE_INSTANCE_FLOATS) {
			this._particleScratch = new Float32Array(
				cappedCount * PARTICLE_INSTANCE_FLOATS
			);
		}

		let writeCount = 0;
		for (let i = 0; i < cappedCount; i++) {
			const particle = particles[i];
			if (!particle) continue;

			const x = particle.position?.x;
			const y = particle.position?.y;
			const z = particle.position?.z;
			const size = particle.size;
			const rotation = particle.rotation;
			if (
				!Number.isFinite(x) ||
				!Number.isFinite(y) ||
				!Number.isFinite(z) ||
				!Number.isFinite(size) ||
				!Number.isFinite(rotation)
			) {
				continue;
			}

			const safeSize = Math.max(0, size);
			if (safeSize <= 0) continue;

			const color = particle.color;
			if (!color) continue;
			const alpha = clamp(Number.isFinite(color.a) ? color.a : 0, 0, 1);
			if (alpha <= 0) continue;
			const red = clamp((Number.isFinite(color.r) ? color.r : 0) / 255, 0, 1);
			const green = clamp(
				(Number.isFinite(color.g) ? color.g : 0) / 255,
				0,
				1
			);
			const blue = clamp((Number.isFinite(color.b) ? color.b : 0) / 255, 0, 1);

			const uvRect = particle.uvRect;
			const u0 = Number.isFinite(uvRect?.u0) ? uvRect.u0 : 0;
			const v0 = Number.isFinite(uvRect?.v0) ? uvRect.v0 : 0;
			const u1 = Number.isFinite(uvRect?.u1) ? uvRect.u1 : 1;
			const v1 = Number.isFinite(uvRect?.v1) ? uvRect.v1 : 1;

			const offset = writeCount * PARTICLE_INSTANCE_FLOATS;
			this._particleScratch[offset] = x;
			this._particleScratch[offset + 1] = y;
			this._particleScratch[offset + 2] = z;
			this._particleScratch[offset + 3] = safeSize;
			this._particleScratch[offset + 4] = red;
			this._particleScratch[offset + 5] = green;
			this._particleScratch[offset + 6] = blue;
			this._particleScratch[offset + 7] = alpha;
			this._particleScratch[offset + 8] = u0;
			this._particleScratch[offset + 9] = v0;
			this._particleScratch[offset + 10] = u1;
			this._particleScratch[offset + 11] = v1;
			this._particleScratch[offset + 12] = rotation;
			writeCount++;
		}

		return writeCount;
	}

	private _ensureParticleResources(): void {
		if (this._particleVao && this._particleQuadBuffer && this._particleInstanceBuffer) {
			return;
		}

		const gl = this._gl;
		const vao = gl.createVertexArray();
		const quadBuffer = gl.createBuffer();
		const instanceBuffer = gl.createBuffer();
		if (!vao || !quadBuffer || !instanceBuffer) {
			if (vao) gl.deleteVertexArray(vao);
			if (quadBuffer) gl.deleteBuffer(quadBuffer);
			if (instanceBuffer) gl.deleteBuffer(instanceBuffer);
			this._warn(
				"webgl-particle-buffer-allocation",
				"Failed to allocate WebGL particle buffers; particle rendering is disabled for this frame"
			);
			return;
		}

		this._particleVao = vao;
		this._particleQuadBuffer = quadBuffer;
		this._particleInstanceBuffer = instanceBuffer;
		this._particleInstanceCapacity = PARTICLE_INITIAL_CAPACITY;
		this._particleScratch = new Float32Array(
			PARTICLE_INITIAL_CAPACITY * PARTICLE_INSTANCE_FLOATS
		);

		gl.bindVertexArray(vao);

		gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, PARTICLE_QUAD_VERTICES, gl.STATIC_DRAW);
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 2, gl.FLOAT, false, PARTICLE_QUAD_STRIDE, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 2, gl.FLOAT, false, PARTICLE_QUAD_STRIDE, 8);

		gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
		gl.bufferData(
			gl.ARRAY_BUFFER,
			this._particleInstanceCapacity * PARTICLE_INSTANCE_STRIDE,
			gl.DYNAMIC_DRAW
		);
		this._bindParticleInstanceAttributes();

		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
	}

	private _ensureParticleCapacity(requiredInstances: number): void {
		if (!this._particleInstanceBuffer || !this._particleVao) return;
		if (requiredInstances <= this._particleInstanceCapacity) return;

		const nextCapacity = Math.max(
			this._particleInstanceCapacity,
			1 << Math.ceil(Math.log2(Math.max(1, requiredInstances)))
		);
		const gl = this._gl;
		const newBuffer = gl.createBuffer();
		if (!newBuffer) {
			this._warn(
				"webgl-particle-buffer-grow",
				`Failed to grow WebGL particle instance buffer to ${nextCapacity}; keeping previous capacity`
			);
			return;
		}

		gl.deleteBuffer(this._particleInstanceBuffer);
		this._particleInstanceBuffer = newBuffer;
		this._particleInstanceCapacity = nextCapacity;
		if (this._particleScratch.length < nextCapacity * PARTICLE_INSTANCE_FLOATS) {
			this._particleScratch = new Float32Array(
				nextCapacity * PARTICLE_INSTANCE_FLOATS
			);
		}

		gl.bindVertexArray(this._particleVao);
		gl.bindBuffer(gl.ARRAY_BUFFER, newBuffer);
		gl.bufferData(
			gl.ARRAY_BUFFER,
			nextCapacity * PARTICLE_INSTANCE_STRIDE,
			gl.DYNAMIC_DRAW
		);
		this._bindParticleInstanceAttributes();
		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
	}

	private _bindParticleInstanceAttributes(): void {
		const gl = this._gl;
		if (!this._particleInstanceBuffer) return;
		gl.bindBuffer(gl.ARRAY_BUFFER, this._particleInstanceBuffer);

		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 0);
		gl.vertexAttribDivisor(2, 1);

		gl.enableVertexAttribArray(3);
		gl.vertexAttribPointer(3, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 16);
		gl.vertexAttribDivisor(3, 1);

		gl.enableVertexAttribArray(4);
		gl.vertexAttribPointer(4, 4, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 32);
		gl.vertexAttribDivisor(4, 1);

		gl.enableVertexAttribArray(5);
		gl.vertexAttribPointer(5, 1, gl.FLOAT, false, PARTICLE_INSTANCE_STRIDE, 48);
		gl.vertexAttribDivisor(5, 1);
	}

	private _destroyParticleResources(): void {
		const gl = this._gl;
		if (this._particleVao) {
			gl.deleteVertexArray(this._particleVao);
			this._particleVao = null;
		}
		if (this._particleQuadBuffer) {
			gl.deleteBuffer(this._particleQuadBuffer);
			this._particleQuadBuffer = null;
		}
		if (this._particleInstanceBuffer) {
			gl.deleteBuffer(this._particleInstanceBuffer);
			this._particleInstanceBuffer = null;
		}
		this._particleInstanceCapacity = 0;
		this._particleScratch = new Float32Array(0);
	}

	private _bindGlobalUniforms(
		sceneProgram: WebGLSceneProgram,
		context: FrameContext
	): void {
		const gl = this._gl;
		const uniforms = sceneProgram.uniforms;
		const lights = this._lightState ?? {
			ambientColor: [0, 0, 0] as [number, number, number],
			directionalLights: [],
			directionalShadows: [],
			pointLights: [],
			spotLights: [],
			spotShadows: [],
		};

		if (uniforms.viewProjection) {
			gl.uniformMatrix4fv(
				uniforms.viewProjection,
				false,
				toColumnMajorMat4(context.camera.viewProjectionMatrix)
			);
		}
		if (uniforms.cameraPosition) {
			const cameraPosition = context.camera.getWorldPosition();
			gl.uniform3f(
				uniforms.cameraPosition,
				cameraPosition.x,
				cameraPosition.y,
				cameraPosition.z
			);
		}
		if (uniforms.ambientColor) {
			gl.uniform3f(
				uniforms.ambientColor,
				lights.ambientColor[0],
				lights.ambientColor[1],
				lights.ambientColor[2]
			);
		}
		if (uniforms.enableLighting) {
			gl.uniform1i(
				uniforms.enableLighting,
				context.features.enableLighting ? 1 : 0
			);
		}
		const shadowsEnabled =
			context.features.enableShadows &&
			!!this._shadowAtlasTexture &&
			this._shadowAtlasTileSize > 0;
		if (uniforms.enableShadows) {
			gl.uniform1i(uniforms.enableShadows, shadowsEnabled ? 1 : 0);
		}
		if (uniforms.shadowAtlas) {
			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gl.TEXTURE_2D, this._shadowAtlasTexture);
			gl.uniform1i(uniforms.shadowAtlas, 1);
			gl.activeTexture(gl.TEXTURE0);
		}

		if (uniforms.taaJitter) {
			gl.uniform4fv(uniforms.taaJitter, this._taaJitter);
		}
		if (uniforms.prevViewProjection) {
			gl.uniformMatrix4fv(
				uniforms.prevViewProjection,
				false,
				this._prevViewProjection ??
					toColumnMajorMat4(context.camera.viewProjectionMatrix)
			);
		}

		if (uniforms.dirLightCount) {
			gl.uniform1i(uniforms.dirLightCount, lights.directionalLights.length);
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
			);
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
			);
		}
		if (uniforms.dirShadowViewProjection) {
			gl.uniformMatrix4fv(
				uniforms.dirShadowViewProjection,
				false,
				flattenShadowViewProjection(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				)
			);
		}
		if (uniforms.dirShadowParamsA) {
			gl.uniform4fv(
				uniforms.dirShadowParamsA,
				flattenShadowParamsA(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				)
			);
		}
		if (uniforms.dirShadowParamsB) {
			gl.uniform4fv(
				uniforms.dirShadowParamsB,
				flattenShadowParamsB(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				)
			);
		}

		if (uniforms.pointLightCount) {
			gl.uniform1i(uniforms.pointLightCount, lights.pointLights.length);
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
			);
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
			);
		}

		if (uniforms.spotLightCount) {
			gl.uniform1i(uniforms.spotLightCount, lights.spotLights.length);
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
			);
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
			);
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
			);
		}
		if (uniforms.spotShadowViewProjection) {
			gl.uniformMatrix4fv(
				uniforms.spotShadowViewProjection,
				false,
				flattenShadowViewProjection(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS)
			);
		}
		if (uniforms.spotShadowParamsA) {
			gl.uniform4fv(
				uniforms.spotShadowParamsA,
				flattenShadowParamsA(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS)
			);
		}
		if (uniforms.spotShadowParamsB) {
			gl.uniform4fv(
				uniforms.spotShadowParamsB,
				flattenShadowParamsB(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS)
			);
		}
	}

	private _renderSkybox(context: FrameContext): void {
		const skyboxTexture = context.scene.skybox;
		if (!skyboxTexture || !this._fullscreenVao) return;

		const gl = this._gl;
		const skyboxProgram = this._programs.getSkyboxProgram();
		const resolved = this._textures.getSkyboxTexture(skyboxTexture);
		const view = context.camera.viewMatrix.elements;
		const isOrthographic = context.camera.type === CameraType.Orthographic;
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360);
		const aspect = context.camera.aspectRatio || this._width / this._height;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		gl.useProgram(skyboxProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.BLEND);
		gl.disable(gl.DEPTH_TEST);
		gl.depthMask(false);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		if (skyboxProgram.uniforms.skyboxMap) {
			gl.uniform1i(skyboxProgram.uniforms.skyboxMap, 0);
		}
		if (skyboxProgram.uniforms.skyboxBasisRight) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisRight,
				view[0][0],
				view[0][1],
				view[0][2],
				tanHalfFov
			);
		}
		if (skyboxProgram.uniforms.skyboxBasisUp) {
			gl.uniform4f(
				skyboxProgram.uniforms.skyboxBasisUp,
				view[1][0],
				view[1][1],
				view[1][2],
				aspect
			);
		}
		if (skyboxProgram.uniforms.skyboxBasisBackward) {
			gl.uniform3f(
				skyboxProgram.uniforms.skyboxBasisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		}
		if (skyboxProgram.uniforms.skyboxIsOrthographic) {
			gl.uniform1f(
				skyboxProgram.uniforms.skyboxIsOrthographic,
				isOrthographic ? 1 : 0
			);
		}
		if (skyboxProgram.uniforms.skyboxMapIsLinear) {
			gl.uniform1i(
				skyboxProgram.uniforms.skyboxMapIsLinear,
				resolved.isLinear ? 1 : 0
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		gl.depthMask(true);
		gl.enable(gl.DEPTH_TEST);
		gl.bindVertexArray(null);
	}

	private _applyFXAA(): void {
		if (!this._postFramebuffer || !this._postColorTexture) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}

		const gl = this._gl;
		const fxaaProgram = this._programs.getFXAAProgram();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			this._postColorTexture,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(fxaaProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (fxaaProgram.uniforms.sourceMap) {
			gl.uniform1i(fxaaProgram.uniforms.sourceMap, 0);
		}
		if (fxaaProgram.uniforms.texelSize) {
			gl.uniform2f(
				fxaaProgram.uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

		this._presentSourceTexture = this._postColorTexture;
	}

	private _applyTAA(options?: TAAOptions): void {
		if (
			!this._sceneColorTexture ||
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._taaHistoryTextures[0] ||
			!this._fullscreenVao
		) {
			return;
		}

		const gl = this._gl;
		const taaProgram = this._programs.getTAAProgram();
		const historyIndex = this._taaHistoryIndex;
		const currentHistory = this._taaHistoryTextures[historyIndex];
		const nextHistory = this._taaHistoryTextures[1 - historyIndex];
		const currentMotionHistory = this._taaMotionHistoryTextures[historyIndex];
		const nextMotionHistory = this._taaMotionHistoryTextures[1 - historyIndex];

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			nextHistory!,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			nextMotionHistory!,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);

		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(taaProgram.program);
		gl.bindVertexArray(this._fullscreenVao);

		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneColorTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, currentHistory!);
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		gl.activeTexture(gl.TEXTURE3);
		gl.bindTexture(gl.TEXTURE_2D, currentMotionHistory!);

		const uniforms = taaProgram.uniforms;
		if (uniforms.sceneColor) gl.uniform1i(uniforms.sceneColor, 0);
		if (uniforms.historyMap) gl.uniform1i(uniforms.historyMap, 1);
		if (uniforms.motionMap) gl.uniform1i(uniforms.motionMap, 2);
		if (uniforms.motionHistory) gl.uniform1i(uniforms.motionHistory, 3);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}

		// Default options
		const weight = (options?.historyWeight as number) ?? 0.9;
		const depthThreshold = (options?.disocclusionDepthThreshold as number) ?? 0.01;
		const motionFactor = (options?.motionFactor as number) ?? 0.1;
		const varianceClampGamma = (options?.varianceClampGamma as number) ?? 1.0;
		const sharpen = (options?.sharpen as number) ?? 0.0;

		if (uniforms.historyWeight) gl.uniform1f(uniforms.historyWeight, weight);
		if (uniforms.depthThreshold)
			gl.uniform1f(uniforms.depthThreshold, depthThreshold);
		if (uniforms.motionFactor) gl.uniform1f(uniforms.motionFactor, motionFactor);
		if (uniforms.varianceClampGamma)
			gl.uniform1f(uniforms.varianceClampGamma, varianceClampGamma);
		if (uniforms.sharpen) gl.uniform1f(uniforms.sharpen, sharpen);
		if (uniforms.historyValid) gl.uniform1f(uniforms.historyValid, 1.0);

		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);

		this._taaHistoryIndex = 1 - historyIndex;
		this._presentSourceTexture = nextHistory;
	}

	private _present(applyGamma: boolean): void {
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture || !this._fullscreenVao) return;
		const gl = this._gl;
		const presentProgram = this._programs.getPresentProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(presentProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (presentProgram.uniforms.sourceMap) {
			gl.uniform1i(presentProgram.uniforms.sourceMap, 0);
		}
		if (presentProgram.uniforms.applyGamma) {
			gl.uniform1i(presentProgram.uniforms.applyGamma, applyGamma ? 1 : 0);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);
		this._presentedInFrame = true;
	}

	private _ensureFrameTargets(width: number, height: number): void {
		if (
			this._sceneFramebuffer &&
			this._sceneColorTexture &&
			this._sceneDepthBuffer &&
			this._postFramebuffer &&
			this._postColorTexture &&
			this._targetWidth === width &&
			this._targetHeight === height
		) {
			return;
		}

		if (
			width > this._maxTextureSize ||
			height > this._maxTextureSize ||
			width > this._maxRenderbufferSize ||
			height > this._maxRenderbufferSize
		) {
			throw new Error(
				`WebGL frame size ${width}x${height} exceeds device limits (MAX_TEXTURE_SIZE=${this._maxTextureSize}, MAX_RENDERBUFFER_SIZE=${this._maxRenderbufferSize})`
			);
		}

		this._destroyFrameTargets();
		const gl = this._gl;

		const supportsFloatColorBuffer = !!gl.getExtension("EXT_color_buffer_float");
		const colorInternalFormat = supportsFloatColorBuffer ? gl.RGBA16F : gl.RGBA8;
		const colorType = supportsFloatColorBuffer ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
		const motionInternalFormat = supportsFloatColorBuffer ? gl.RGBA16F : gl.RGBA8;
		const motionType = supportsFloatColorBuffer ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
		if (!supportsFloatColorBuffer) {
			this._warn(
				"webgl-motion-float-unsupported",
				"EXT_color_buffer_float is unavailable; falling back to RGBA8 motion attachments."
			);
		}

		const sceneFramebuffer = gl.createFramebuffer();
		const sceneColorTexture = this._createColorTexture(
			width,
			height,
			colorInternalFormat,
			colorType
		);
		const sceneMotionTexture = this._createColorTexture(
			width,
			height,
			motionInternalFormat,
			motionType
		);
		const sceneDepthBuffer = gl.createRenderbuffer();
		const postFramebuffer = gl.createFramebuffer();
		const postColorTexture = this._createColorTexture(
			width,
			height,
			colorInternalFormat,
			colorType
		);

		const history0 = this._createColorTexture(
			width,
			height,
			colorInternalFormat,
			colorType
		);
		const history1 = this._createColorTexture(
			width,
			height,
			colorInternalFormat,
			colorType
		);
		const motionHistory0 = this._createColorTexture(
			width,
			height,
			motionInternalFormat,
			motionType
		);
		const motionHistory1 = this._createColorTexture(
			width,
			height,
			motionInternalFormat,
			motionType
		);

		const cleanupAllocatedTargets = (): void => {
			if (sceneFramebuffer) gl.deleteFramebuffer(sceneFramebuffer);
			if (sceneColorTexture) gl.deleteTexture(sceneColorTexture);
			if (sceneMotionTexture) gl.deleteTexture(sceneMotionTexture);
			if (sceneDepthBuffer) gl.deleteRenderbuffer(sceneDepthBuffer);
			if (postFramebuffer) gl.deleteFramebuffer(postFramebuffer);
			if (postColorTexture) gl.deleteTexture(postColorTexture);
			if (history0) gl.deleteTexture(history0);
			if (history1) gl.deleteTexture(history1);
			if (motionHistory0) gl.deleteTexture(motionHistory0);
			if (motionHistory1) gl.deleteTexture(motionHistory1);
		};

		if (
			!sceneFramebuffer ||
			!sceneColorTexture ||
			!sceneMotionTexture ||
			!sceneDepthBuffer ||
			!postFramebuffer ||
			!postColorTexture ||
			!history0 ||
			!history1 ||
			!motionHistory0 ||
			!motionHistory1
		) {
			cleanupAllocatedTargets();
			throw new Error("Failed to create WebGL frame targets");
		}

		gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepthBuffer);
		gl.renderbufferStorage(
			gl.RENDERBUFFER,
			gl.DEPTH_COMPONENT24,
			width,
			height
		);
		gl.bindRenderbuffer(gl.RENDERBUFFER, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			sceneColorTexture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			sceneMotionTexture,
			0
		);
		gl.framebufferRenderbuffer(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.RENDERBUFFER,
			sceneDepthBuffer
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
			cleanupAllocatedTargets();
			throw new Error(
				`WebGL scene framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		gl.bindFramebuffer(gl.FRAMEBUFFER, postFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			postColorTexture,
			0
		);
		status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			cleanupAllocatedTargets();
			throw new Error(
				`WebGL post framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		this._sceneFramebuffer = sceneFramebuffer;
		this._sceneColorTexture = sceneColorTexture;
		this._sceneMotionTexture = sceneMotionTexture;
		this._sceneDepthBuffer = sceneDepthBuffer;
		this._taaHistoryTextures = [history0, history1];
		this._taaMotionHistoryTextures = [motionHistory0, motionHistory1];
		this._taaHistoryIndex = 0;
		this._postFramebuffer = postFramebuffer;
		this._postColorTexture = postColorTexture;
		this._presentSourceTexture = sceneColorTexture;
		this._targetWidth = width;
		this._targetHeight = height;
	}

	private _destroyFrameTargets(): void {
		const gl = this._gl;
		if (this._sceneFramebuffer) {
			gl.deleteFramebuffer(this._sceneFramebuffer);
			this._sceneFramebuffer = null;
		}
		if (this._sceneColorTexture) {
			gl.deleteTexture(this._sceneColorTexture);
			this._sceneColorTexture = null;
		}
		if (this._sceneMotionTexture) {
			gl.deleteTexture(this._sceneMotionTexture);
			this._sceneMotionTexture = null;
		}
		if (this._sceneDepthBuffer) {
			gl.deleteRenderbuffer(this._sceneDepthBuffer);
			this._sceneDepthBuffer = null;
		}
		for (const texture of this._taaHistoryTextures) {
			if (texture) gl.deleteTexture(texture);
		}
		for (const texture of this._taaMotionHistoryTextures) {
			if (texture) gl.deleteTexture(texture);
		}
		this._taaHistoryTextures = [null, null];
		this._taaMotionHistoryTextures = [null, null];
		if (this._postFramebuffer) {
			gl.deleteFramebuffer(this._postFramebuffer);
			this._postFramebuffer = null;
		}
		if (this._postColorTexture) {
			gl.deleteTexture(this._postColorTexture);
			this._postColorTexture = null;
		}
		this._presentSourceTexture = null;
		this._targetWidth = 0;
		this._targetHeight = 0;
	}

	private _pruneModelMatrixCache(): void {
		if (this._modelMatrixCache.size <= this._modelMatrixKeysThisFrame.size) {
			return;
		}
		for (const cacheKey of this._modelMatrixCache.keys()) {
			if (!this._modelMatrixKeysThisFrame.has(cacheKey)) {
				this._modelMatrixCache.delete(cacheKey);
			}
		}
	}

	private _createColorTexture(
		width: number,
		height: number,
		internalFormat: number = this._gl.RGBA8,
		type: number = this._gl.UNSIGNED_BYTE
	): WebGLTexture | null {
		const gl = this._gl;
		const texture = gl.createTexture();
		if (!texture) return null;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		
		let format = gl.RGBA;
		if (internalFormat === gl.RGBA16F || internalFormat === gl.RGBA32F) {
			format = gl.RGBA;
		}
		
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			width,
			height,
			0,
			format,
			type,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);
		return texture;
	}

	private _resolveLimit(parameter: number, fallback: number): number {
		try {
			const value = this._gl.getParameter(parameter);
			if (typeof value === "number" && Number.isFinite(value) && value > 0) {
				return Math.floor(value);
			}
		} catch {}
		return fallback;
	}

	private _setCullMode(material: Material): void {
		const gl = this._gl;
		if (material.doubleSided || material.cullMode === "none") {
			gl.disable(gl.CULL_FACE);
			return;
		}
		gl.enable(gl.CULL_FACE);
		gl.frontFace(gl.CCW);
		if (material.cullMode === "front") {
			gl.cullFace(gl.FRONT);
		} else {
			gl.cullFace(gl.BACK);
		}
	}
}

function resolveMaterialUniforms(material: Material): MaterialUniformState {
	const isPBR =
		material.shading === ShadingModel.PBR || material.type === "PBR";
	const isUnlit = material.shading === ShadingModel.Unlit;

	let baseColor: [number, number, number] = [1, 1, 1];
	let emissive: [number, number, number] = [0, 0, 0];
	let roughness = 0.5;
	let metalness = 0;
	let reflectance = 0.5;
	let shininess = 32;
	let baseMap: any | null = material.map ?? null;

	if (isPBR) {
		const pbr = material as any;
		const albedo = pbr.albedo ?? { r: 255, g: 255, b: 255 };
		baseColor = [
			clamp((albedo.r ?? 255) / 255, 0, 1),
			clamp((albedo.g ?? 255) / 255, 0, 1),
			clamp((albedo.b ?? 255) / 255, 0, 1),
		];
		const emissiveColor = pbr.emissive ?? { r: 0, g: 0, b: 0 };
		const emissiveIntensity = clamp(pbr.emissiveIntensity ?? 1, 0, 64);
		emissive = [
			clamp((emissiveColor.r ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.g ?? 0) / 255, 0, 1) * emissiveIntensity,
			clamp((emissiveColor.b ?? 0) / 255, 0, 1) * emissiveIntensity,
		];
		roughness = clamp(pbr.roughness ?? 0.5, 0.04, 1);
		metalness = clamp(pbr.metalness ?? 0, 0, 1);
		reflectance = clamp(pbr.reflectance ?? 0.5, 0, 1);
		baseMap = pbr.map ?? baseMap;
	} else {
		const basic = material as any;
		const diffuse = basic.diffuse ?? { r: 255, g: 255, b: 255 };
		baseColor = [
			sRGBToLinear(clamp((diffuse.r ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.g ?? 255) / 255, 0, 1)),
			sRGBToLinear(clamp((diffuse.b ?? 255) / 255, 0, 1)),
		];
		const emissiveColor = basic.emissive;
		if (emissiveColor) {
			const emissiveIntensity = clamp(basic.emissiveIntensity ?? 1, 0, 64);
			emissive = [
				sRGBToLinear(clamp((emissiveColor.r ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.g ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
				sRGBToLinear(clamp((emissiveColor.b ?? 0) / 255, 0, 1)) *
					emissiveIntensity,
			];
		}
		shininess = Math.max(1, basic.shininess ?? 32);
	}

	const opacity = clamp(material.opacity ?? 1, 0, 1);
	const alphaCutoff = clamp(material.alphaCutoff ?? 0.5, 0, 1);
	const alphaModeMask = material.alphaMode === AlphaMode.Mask ? 1 : 0;

	return {
		shadingModel:
			isUnlit ? 2
			: isPBR ? 1
			: 0,
		baseColor: [baseColor[0], baseColor[1], baseColor[2], opacity],
		emissive,
		pbr: [roughness, metalness, reflectance, 0],
		phong: [shininess, 0, 0, 0],
		alpha: [alphaCutoff, alphaModeMask, 0, 0],
		baseMap,
	};
}

function flattenVec4<T>(
	values: T[],
	mapper: (value: T) => [number, number, number, number]
): Float32Array {
	const packed = new Float32Array(16);
	const count = Math.min(4, values.length);
	for (let i = 0; i < count; i++) {
		const value = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = value[0];
		packed[offset + 1] = value[1];
		packed[offset + 2] = value[2];
		packed[offset + 3] = value[3];
	}
	return packed;
}

function flattenShadowViewProjection(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 16);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i]?.viewProjectionMatrix;
		if (!matrix) {
			continue;
		}
		packed.set(toColumnMajorMat4(matrix), i * 16);
	}
	return packed;
}

function flattenShadowParamsA(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = shadow.enabled ? 1 : 0;
		packed[offset + 1] = shadow.depthBias;
		packed[offset + 2] = shadow.normalBias;
		packed[offset + 3] = shadow.normalBiasMin;
	}
	return packed;
}

function flattenShadowParamsB(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = shadow.pcfRadius;
		packed[offset + 1] = shadow.shadowStrength;
		packed[offset + 2] = shadow.shadowMapSize;
		packed[offset + 3] = shadow.atlasTileSize;
	}
	return packed;
}

function getMaxShadowSize(values: WebGLShadowData[]): number {
	let maxSize = 0;
	for (const shadow of values) {
		if (!shadow.enabled || !shadow.shadowMap) continue;
		maxSize = Math.max(maxSize, shadow.shadowMapSize | 0);
	}
	return maxSize;
}

function toColumnMajorMat4(matrix: Matrix4 | number[][]): Float32Array {
	const elements = matrix instanceof Array ? matrix : matrix.elements;
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
	]);
}

function toColumnMajorMat3(matrix: Matrix4 | Matrix3Arr): Float32Array | null {
	const rows: number[][] =
		matrix instanceof Array ? matrix : (matrix as Matrix4).elements;
	if (!rows || rows.length < 3) return null;
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
	];
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return new Float32Array(values);
}

function isFiniteMatrix(matrix: Matrix4): boolean {
	const elements = matrix.elements;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			if (!Number.isFinite(elements[row][col])) {
				return false;
			}
		}
	}
	return true;
}

function toSafeDimension(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}

function resolveTextureUVTransform(texture: any | null): {
	repeatX: number;
	repeatY: number;
	offsetX: number;
	offsetY: number;
	cosRotation: number;
	sinRotation: number;
} {
	const repeatX =
		Number.isFinite(texture?.repeat?.x) ? Math.max(0, texture.repeat.x) : 1;
	const repeatY =
		Number.isFinite(texture?.repeat?.y) ? Math.max(0, texture.repeat.y) : 1;
	const offsetX = Number.isFinite(texture?.offset?.x) ? texture.offset.x : 0;
	const offsetY = Number.isFinite(texture?.offset?.y) ? texture.offset.y : 0;
	const rotation = Number.isFinite(texture?.rotation) ? texture.rotation : 0;
	return {
		repeatX,
		repeatY,
		offsetX,
		offsetY,
		cosRotation: Math.cos(rotation),
		sinRotation: Math.sin(rotation),
	};
}
