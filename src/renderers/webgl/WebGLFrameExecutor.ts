import { CameraType } from "../../cameras/Camera";
import { isShadowCastingLight } from "../../lights";
import { ParticleBlendMode } from "../../particles";
import {
	AlphaMode,
	ShadingModel,
	type Material,
} from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
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
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	type DrawPacket,
	type BloomOptions,
	type DOFOptions,
	type FrameContext,
	type FramePass,
	type MotionBlurOptions,
	type ParticleRenderBatch,
	type SSAOOptions,
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
import type { ShaderRuntime } from "../../shaders/runtime";
import type { ShaderCompileError } from "../../shaders/runtime";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";

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
	"ssao",
	"motion-blur",
	"dof",
	"bloom",
	"fxaa",
	"taa",
	"gamma",
]);

const TAA_HALTON_SAMPLE_COUNT = 16;
const TAA_HISTORY_WEIGHT_RANGE: [number, number] = [0, 0.99];
const TAA_DEPTH_THRESHOLD_RANGE: [number, number] = [1e-4, 1];
const TAA_MOTION_FACTOR_RANGE: [number, number] = [0, 512];
const TAA_VARIANCE_GAMMA_RANGE: [number, number] = [0, 8];
const TAA_SHARPEN_RANGE: [number, number] = [0, 2];
const MOTION_BLUR_SHUTTER_SCALE_RANGE: [number, number] = [0, 2];
const MOTION_BLUR_MAX_SAMPLES_RANGE: [number, number] = [4, 64];
const MOTION_BLUR_VELOCITY_CLAMP_RANGE: [number, number] = [0.005, 0.25];
const MOTION_BLUR_DEPTH_REJECT_RANGE: [number, number] = [0.0001, 0.25];
const MOTION_BLUR_CENTER_WEIGHT_RANGE: [number, number] = [0, 4];
const DOF_NEAR_FAR_STRENGTH_RANGE: [number, number] = [0, 2];
const DOF_MAX_BLUR_RADIUS_RANGE: [number, number] = [0, 32];
const DOF_DEPTH_CURVE_RANGE: [number, number] = [0.25, 4];
const DOF_HIGHLIGHT_GAIN_RANGE: [number, number] = [0, 3];
const DOF_CHROMATIC_ABERRATION_RANGE: [number, number] = [0, 2];
const IDENTITY_MATRIX4_COLUMN_MAJOR = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
]);

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
	private _sceneNormalTexture: WebGLTexture | null = null;
	private _sceneDepthBuffer: WebGLRenderbuffer | null = null;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();
	private _taaHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaMotionHistoryTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
	private _taaHistoryIndex = 0;
	private _taaHistoryValid = false;
	private _taaJitter = new Float32Array(4); // currX, currY, prevX, prevY
	private _taaFrameIndex = 0;
	private _prevViewProjection: Float32Array | null = null;
	private _modelMatrixCache = new Map<string, Float32Array>();
	private _modelMatrixKeysThisFrame = new Set<string>();
	private _postFramebuffer: WebGLFramebuffer | null = null;
	private _postColorTexture: WebGLTexture | null = null;
	private _ssaoRawTexture: WebGLTexture | null = null;
	private _ssaoBlurTexture: WebGLTexture | null = null;
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
	private _targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
	private _maxTextureSize: number;
	private _maxRenderbufferSize: number;
	private _presentedInFrame = false;
	private _activeContext: FrameContext | null = null;
	private _lightState: WebGLLightState | null = null;
	private _ssaoFrameIndex = 0;

	constructor(
		gl: WebGL2RenderingContext,
		warn: WarnFn,
		shaderRuntime?: ShaderRuntime
	) {
		this._gl = gl;
		this._warn = warn;
		this._programs = new WebGLProgramLibrary(gl, warn, shaderRuntime);
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
		const ssaoDownsample = clampDownsample(
			context.features.ssaoOptions?.downsample,
			DEFAULT_SSAO_OPTIONS.downsample
		);
		this._ensureFrameTargets(this._width, this._height, ssaoDownsample);
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
			this._taaHistoryValid = false;
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
			case "ssao":
				this._applySSAO(context.features.ssaoOptions, context);
				break;
			case "motion-blur":
				this._applyMotionBlur(context.features.motionBlurOptions);
				break;
			case "dof":
				this._applyDOF(context.features.dofOptions);
				break;
			case "bloom":
				this._applyBloom(context.features.bloomOptions);
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

	public warmup(
		context: FrameContext,
		plan: WarmupPlan
	): WarmupPhaseCounters {
		let total = 0;
		let compiled = 0;
		let skipped = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];

		const compile = (label: string, action: () => void): void => {
			total++;
			try {
				action();
				compiled++;
			} catch (error) {
				failed++;
				errors.push(toShaderCompileError(error, "webgl", label));
			}
		};

		compile("WebGLSceneProgram:builtin", () => {
			this._programs.getSceneProgram();
		});
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) {
				continue;
			}
			compile(`WebGLSceneProgram:material:${material.shaderId}`, () => {
				this._programs.getSceneProgram(material);
			});
		}

		if (plan.enableSkybox) {
			compile("WebGLSkyboxProgram", () => {
				this._programs.getSkyboxProgram();
			});
		}
		if (plan.enableShadows) {
			compile("WebGLShadowDepthProgram", () => {
				this._programs.getShadowDepthProgram();
			});
		}
		if (plan.enableParticles) {
			compile("WebGLParticleProgram", () => {
				this._programs.getParticleProgram();
			});
		}

		for (const pass of plan.postProcessPasses) {
			switch (pass) {
				case "ssao":
					compile("WebGLSSAORawProgram", () => {
						this._programs.getSSAORawProgram();
					});
					compile("WebGLSSAOBlurProgram", () => {
						this._programs.getSSAOBlurProgram();
					});
					compile("WebGLSSAOCombineProgram", () => {
						this._programs.getSSAOCombineProgram();
					});
					break;
				case "taa":
					compile("WebGLTAAProgram", () => {
						this._programs.getTAAProgram();
					});
					break;
				case "fxaa":
					compile("WebGLFXAAProgram", () => {
						this._programs.getFXAAProgram();
					});
					break;
				case "bloom":
					compile("WebGLBloomProgram", () => {
						this._programs.getBloomProgram();
					});
					break;
				case "motion-blur":
					compile("WebGLMotionBlurProgram", () => {
						this._programs.getMotionBlurProgram();
					});
					break;
				case "dof":
					compile("WebGLDOFProgram", () => {
						this._programs.getDOFProgram();
					});
					break;
				case "gamma":
					compile("WebGLPresentProgram", () => {
						this._programs.getPresentProgram();
					});
					break;
				case "ssr":
					compile("WebGLSSRProgram", () => {
						this._programs.getSSRProgram();
					});
					break;
				case "volumetric":
					compile("WebGLVolumetricProgram", () => {
						this._programs.getVolumetricProgram();
					});
					break;
				default:
					skipped++;
					break;
			}
		}

		if (context.features.enableGamma && !plan.postProcessPasses.includes("gamma")) {
			compile("WebGLPresentProgram", () => {
				this._programs.getPresentProgram();
			});
		}

		return {
			phase: "webgl-programs",
			total,
			compiled,
			skipped,
			failed,
			errors,
		};
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
		if (!transparent && this._sceneNormalTexture) {
			gl.drawBuffers([
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
			]);
		} else {
			gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		}
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

		const currentViewProjection = toFiniteColumnMajorMat4(
			context.camera.viewProjectionMatrix
		);
		if (!currentViewProjection) {
			this._warn(
				"webgl-camera-view-projection-invalid",
				"WebGL camera view-projection matrix is non-finite; resetting temporal history."
			);
			this._prevViewProjection = null;
			this._taaHistoryValid = false;
		} else if (this._prevViewProjection) {
			this._prevViewProjection.set(currentViewProjection);
		} else {
			this._prevViewProjection = currentViewProjection;
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
			const viewProjection = toFiniteColumnMajorMat4(
				context.camera.viewProjectionMatrix
			);
			if (!viewProjection) {
				this._warn(
					"webgl-camera-view-projection-invalid",
					"WebGL camera view-projection matrix is non-finite; using identity matrix."
				);
			}
			gl.uniformMatrix4fv(
				uniforms.viewProjection,
				false,
				viewProjection ?? IDENTITY_MATRIX4_COLUMN_MAJOR
			);
		}
		if (uniforms.viewMatrix) {
			const viewMatrix = toFiniteColumnMajorMat4(context.camera.viewMatrix);
			if (!viewMatrix) {
				this._warn(
					"webgl-camera-view-matrix-invalid",
					"WebGL camera view matrix is non-finite; using identity matrix."
				);
			}
			gl.uniformMatrix4fv(
				uniforms.viewMatrix,
				false,
				viewMatrix ?? IDENTITY_MATRIX4_COLUMN_MAJOR
			);
		}
		if (uniforms.cameraPosition) {
			const cameraPosition = context.camera.getWorldPosition();
			const cameraX = finiteOr(cameraPosition.x, 0);
			const cameraY = finiteOr(cameraPosition.y, 0);
			const cameraZ = finiteOr(cameraPosition.z, 0);
			if (
				cameraX !== cameraPosition.x ||
				cameraY !== cameraPosition.y ||
				cameraZ !== cameraPosition.z
			) {
				this._warn(
					"webgl-camera-position-invalid",
					"WebGL camera position is non-finite; using origin fallback."
				);
			}
			gl.uniform3f(
				uniforms.cameraPosition,
				cameraX,
				cameraY,
				cameraZ
			);
		}
		if (uniforms.ambientColor) {
			const ambientR = finiteOr(lights.ambientColor[0], 0);
			const ambientG = finiteOr(lights.ambientColor[1], 0);
			const ambientB = finiteOr(lights.ambientColor[2], 0);
			if (
				ambientR !== lights.ambientColor[0] ||
				ambientG !== lights.ambientColor[1] ||
				ambientB !== lights.ambientColor[2]
			) {
				this._warn(
					"webgl-ambient-color-invalid",
					"WebGL ambient light color contains non-finite values; using black fallback."
				);
			}
			gl.uniform3f(
				uniforms.ambientColor,
				ambientR,
				ambientG,
				ambientB
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
			const prevViewProjection = sanitizeFloat32Array(
				this._prevViewProjection ??
					toColumnMajorMat4(context.camera.viewProjectionMatrix),
				0
			);
			if (prevViewProjection.hadInvalid) {
				this._warn(
					"webgl-prev-view-projection-invalid",
					"WebGL previous view-projection matrix is non-finite; using sanitized values."
				);
			}
			gl.uniformMatrix4fv(
				uniforms.prevViewProjection,
				false,
				prevViewProjection.values
			);
		}

		if (uniforms.dirLightCount) {
			gl.uniform1i(uniforms.dirLightCount, lights.directionalLights.length);
		}
		if (uniforms.dirLightDirection) {
			const packedDirection = sanitizeFloat32Array(
				flattenVec4(lights.directionalLights, (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					0,
				]),
				0
			);
			if (packedDirection.hadInvalid) {
				this._warn(
					"webgl-dir-light-direction-invalid",
					"WebGL directional light direction contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.dirLightDirection,
				packedDirection.values
			);
		}
		if (uniforms.dirLightColor) {
			const packedColor = sanitizeFloat32Array(
				flattenVec4(lights.directionalLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				]),
				0
			);
			if (packedColor.hadInvalid) {
				this._warn(
					"webgl-dir-light-color-invalid",
					"WebGL directional light color contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.dirLightColor,
				packedColor.values
			);
		}
		if (uniforms.dirShadowViewProjection) {
			const packedShadowViewProjection = sanitizeFloat32Array(
				flattenShadowViewProjection(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				),
				0
			);
			if (packedShadowViewProjection.hadInvalid) {
				this._warn(
					"webgl-dir-shadow-view-projection-invalid",
					"WebGL directional shadow matrix contains non-finite values; using sanitized values."
				);
			}
			gl.uniformMatrix4fv(
				uniforms.dirShadowViewProjection,
				false,
				packedShadowViewProjection.values
			);
		}
		if (uniforms.dirShadowParamsA) {
			const packedDirShadowParamsA = sanitizeFloat32Array(
				flattenShadowParamsA(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				),
				0
			);
			if (packedDirShadowParamsA.hadInvalid) {
				this._warn(
					"webgl-dir-shadow-params-a-invalid",
					"WebGL directional shadow parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.dirShadowParamsA,
				packedDirShadowParamsA.values
			);
		}
		if (uniforms.dirShadowParamsB) {
			const packedDirShadowParamsB = sanitizeFloat32Array(
				flattenShadowParamsB(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				),
				0
			);
			if (packedDirShadowParamsB.hadInvalid) {
				this._warn(
					"webgl-dir-shadow-params-b-invalid",
					"WebGL directional shadow parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.dirShadowParamsB,
				packedDirShadowParamsB.values
			);
		}
		if (uniforms.dirShadowParamsC) {
			const packedDirShadowParamsC = sanitizeFloat32Array(
				flattenShadowParamsC(
					lights.directionalShadows,
					WEBGL_MAX_DIRECTIONAL_LIGHTS
				),
				0
			);
			if (packedDirShadowParamsC.hadInvalid) {
				this._warn(
					"webgl-dir-shadow-params-c-invalid",
					"WebGL directional shadow slope parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.dirShadowParamsC,
				packedDirShadowParamsC.values
			);
		}

		if (uniforms.pointLightCount) {
			gl.uniform1i(uniforms.pointLightCount, lights.pointLights.length);
		}
		if (uniforms.pointLightPositionRange) {
			const packedPointPositionRange = sanitizeFloat32Array(
				flattenVec4(lights.pointLights, (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				]),
				0
			);
			if (packedPointPositionRange.hadInvalid) {
				this._warn(
					"webgl-point-light-position-invalid",
					"WebGL point light position/range contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.pointLightPositionRange,
				packedPointPositionRange.values
			);
		}
		if (uniforms.pointLightColor) {
			const packedPointColor = sanitizeFloat32Array(
				flattenVec4(lights.pointLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				]),
				0
			);
			if (packedPointColor.hadInvalid) {
				this._warn(
					"webgl-point-light-color-invalid",
					"WebGL point light color contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.pointLightColor,
				packedPointColor.values
			);
		}

		if (uniforms.spotLightCount) {
			gl.uniform1i(uniforms.spotLightCount, lights.spotLights.length);
		}
		if (uniforms.spotLightPositionRange) {
			const packedSpotPositionRange = sanitizeFloat32Array(
				flattenVec4(lights.spotLights, (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				]),
				0
			);
			if (packedSpotPositionRange.hadInvalid) {
				this._warn(
					"webgl-spot-light-position-invalid",
					"WebGL spot light position/range contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotLightPositionRange,
				packedSpotPositionRange.values
			);
		}
		if (uniforms.spotLightDirectionOuter) {
			const packedSpotDirectionOuter = sanitizeFloat32Array(
				flattenVec4(lights.spotLights, (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					light.outerCos,
				]),
				0
			);
			if (packedSpotDirectionOuter.hadInvalid) {
				this._warn(
					"webgl-spot-light-direction-invalid",
					"WebGL spot light direction/outer cone contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotLightDirectionOuter,
				packedSpotDirectionOuter.values
			);
		}
		if (uniforms.spotLightColorInner) {
			const packedSpotColorInner = sanitizeFloat32Array(
				flattenVec4(lights.spotLights, (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					light.innerCos,
				]),
				0
			);
			if (packedSpotColorInner.hadInvalid) {
				this._warn(
					"webgl-spot-light-color-invalid",
					"WebGL spot light color/inner cone contains non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotLightColorInner,
				packedSpotColorInner.values
			);
		}
		if (uniforms.spotShadowViewProjection) {
			const packedSpotShadowViewProjection = sanitizeFloat32Array(
				flattenShadowViewProjection(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS),
				0
			);
			if (packedSpotShadowViewProjection.hadInvalid) {
				this._warn(
					"webgl-spot-shadow-view-projection-invalid",
					"WebGL spot shadow matrix contains non-finite values; using sanitized values."
				);
			}
			gl.uniformMatrix4fv(
				uniforms.spotShadowViewProjection,
				false,
				packedSpotShadowViewProjection.values
			);
		}
		if (uniforms.spotShadowParamsA) {
			const packedSpotShadowParamsA = sanitizeFloat32Array(
				flattenShadowParamsA(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS),
				0
			);
			if (packedSpotShadowParamsA.hadInvalid) {
				this._warn(
					"webgl-spot-shadow-params-a-invalid",
					"WebGL spot shadow parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotShadowParamsA,
				packedSpotShadowParamsA.values
			);
		}
		if (uniforms.spotShadowParamsB) {
			const packedSpotShadowParamsB = sanitizeFloat32Array(
				flattenShadowParamsB(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS),
				0
			);
			if (packedSpotShadowParamsB.hadInvalid) {
				this._warn(
					"webgl-spot-shadow-params-b-invalid",
					"WebGL spot shadow parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotShadowParamsB,
				packedSpotShadowParamsB.values
			);
		}
		if (uniforms.spotShadowParamsC) {
			const packedSpotShadowParamsC = sanitizeFloat32Array(
				flattenShadowParamsC(lights.spotShadows, WEBGL_MAX_SPOT_LIGHTS),
				0
			);
			if (packedSpotShadowParamsC.hadInvalid) {
				this._warn(
					"webgl-spot-shadow-params-c-invalid",
					"WebGL spot shadow slope parameters contain non-finite values; using sanitized values."
				);
			}
			gl.uniform4fv(
				uniforms.spotShadowParamsC,
				packedSpotShadowParamsC.values
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

	private _applySSAO(options: SSAOOptions | undefined, context: FrameContext): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._sceneMotionTexture ||
			!this._sceneNormalTexture ||
			!this._ssaoRawTexture ||
			!this._ssaoBlurTexture ||
			!this._fullscreenVao
		) {
			return;
		}

		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const rawProgram = this._programs.getSSAORawProgram();
		const blurProgram = this._programs.getSSAOBlurProgram();
		const combineProgram = this._programs.getSSAOCombineProgram();
		const radius = Math.max(1, finiteOr(options?.radius, DEFAULT_SSAO_OPTIONS.radius));
		const bias = Math.max(1e-4, finiteOr(options?.bias, DEFAULT_SSAO_OPTIONS.bias));
		const intensity = Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_SSAO_OPTIONS.intensity)
		);
		const blurRadius = clamp(
			finiteOr(options?.blurRadius, DEFAULT_SSAO_OPTIONS.blurRadius),
			1,
			4
		);
		const blurSharpness = Math.max(
			1e-3,
			finiteOr(options?.blurSharpness, DEFAULT_SSAO_OPTIONS.blurSharpness)
		);
		const samples = clamp(
			Math.round(finiteOr(options?.samples, DEFAULT_SSAO_OPTIONS.samples)),
			4,
			48
		);
		const isOrthographic = context.camera.type === CameraType.Orthographic;
		const tanHalfFov = isOrthographic ? 0 : Math.tan((context.camera.fov * Math.PI) / 360);
		const aspect =
			context.camera.aspectRatio || this._width / Math.max(this._height, 1);
		const fullInvW = 1 / Math.max(this._width, 1);
		const fullInvH = 1 / Math.max(this._height, 1);
		const aoWidth = Math.max(
			1,
			Math.floor(this._width / Math.max(this._targetSSAODownsample, 1))
		);
		const aoHeight = Math.max(
			1,
			Math.floor(this._height / Math.max(this._targetSSAODownsample, 1))
		);
		const aoInvW = 1 / aoWidth;
		const aoInvH = 1 / aoHeight;
		this._ssaoFrameIndex = (this._ssaoFrameIndex + 1) % 1024;
		const frameJitter = this._ssaoFrameIndex / 1024;
		const view = context.camera.viewMatrix.elements;
		const cameraPosition = context.camera.getWorldPosition();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		this._bindPostSingleColorTarget(this._ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(rawProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneNormalTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (rawProgram.uniforms.normalMap) gl.uniform1i(rawProgram.uniforms.normalMap, 0);
		if (rawProgram.uniforms.depthMap) gl.uniform1i(rawProgram.uniforms.depthMap, 1);
		if (rawProgram.uniforms.invSize)
			gl.uniform4f(rawProgram.uniforms.invSize, fullInvW, fullInvH, aoInvW, aoInvH);
		if (rawProgram.uniforms.gtao)
			gl.uniform4f(rawProgram.uniforms.gtao, radius, bias, intensity, samples);
		if (rawProgram.uniforms.blurProj)
			gl.uniform4f(
				rawProgram.uniforms.blurProj,
				blurRadius,
				blurSharpness,
				tanHalfFov,
				aspect
			);
		if (rawProgram.uniforms.pass)
			gl.uniform4f(
				rawProgram.uniforms.pass,
				1,
				0,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		if (rawProgram.uniforms.cameraPosition)
			gl.uniform3f(
				rawProgram.uniforms.cameraPosition,
				cameraPosition.x,
				cameraPosition.y,
				cameraPosition.z
			);
		if (rawProgram.uniforms.basisRight)
			gl.uniform3f(
				rawProgram.uniforms.basisRight,
				view[0][0],
				view[0][1],
				view[0][2]
			);
		if (rawProgram.uniforms.basisUp)
			gl.uniform3f(rawProgram.uniforms.basisUp, view[1][0], view[1][1], view[1][2]);
		if (rawProgram.uniforms.basisBackward)
			gl.uniform3f(
				rawProgram.uniforms.basisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		this._bindPostSingleColorTarget(this._ssaoBlurTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(blurProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoRawTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (blurProgram.uniforms.sourceMap) gl.uniform1i(blurProgram.uniforms.sourceMap, 0);
		if (blurProgram.uniforms.depthMap) gl.uniform1i(blurProgram.uniforms.depthMap, 1);
		if (blurProgram.uniforms.invSize)
			gl.uniform4f(blurProgram.uniforms.invSize, fullInvW, fullInvH, aoInvW, aoInvH);
		if (blurProgram.uniforms.blurProj)
			gl.uniform4f(
				blurProgram.uniforms.blurProj,
				blurRadius,
				blurSharpness,
				tanHalfFov,
				aspect
			);
		if (blurProgram.uniforms.pass)
			gl.uniform4f(
				blurProgram.uniforms.pass,
				1,
				0,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		this._bindPostSingleColorTarget(this._ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoBlurTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);
		if (blurProgram.uniforms.pass)
			gl.uniform4f(
				blurProgram.uniforms.pass,
				0,
				1,
				isOrthographic ? 1 : 0,
				frameJitter
			);
		gl.drawArrays(gl.TRIANGLES, 0, 3);

		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(combineProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._ssaoRawTexture);
		if (combineProgram.uniforms.sceneColor)
			gl.uniform1i(combineProgram.uniforms.sceneColor, 0);
		if (combineProgram.uniforms.aoMap) gl.uniform1i(combineProgram.uniforms.aoMap, 1);
		if (combineProgram.uniforms.invSize)
			gl.uniform4f(
				combineProgram.uniforms.invSize,
				fullInvW,
				fullInvH,
				aoInvW,
				aoInvH
			);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyMotionBlur(options?: MotionBlurOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const shutterScale = sanitizeFiniteClamped(
			options?.shutterScale,
			DEFAULT_MOTION_BLUR_OPTIONS.shutterScale,
			MOTION_BLUR_SHUTTER_SCALE_RANGE[0],
			MOTION_BLUR_SHUTTER_SCALE_RANGE[1]
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options?.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			MOTION_BLUR_MAX_SAMPLES_RANGE[0],
			MOTION_BLUR_MAX_SAMPLES_RANGE[1]
		);
		const velocityClamp = sanitizeFiniteClamped(
			options?.velocityClamp,
			DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp,
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[0],
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[1]
		);
		const depthReject = sanitizeFiniteClamped(
			options?.depthReject,
			DEFAULT_MOTION_BLUR_OPTIONS.depthReject,
			MOTION_BLUR_DEPTH_REJECT_RANGE[0],
			MOTION_BLUR_DEPTH_REJECT_RANGE[1]
		);
		const centerWeight = sanitizeFiniteClamped(
			options?.centerWeight,
			DEFAULT_MOTION_BLUR_OPTIONS.centerWeight,
			MOTION_BLUR_CENTER_WEIGHT_RANGE[0],
			MOTION_BLUR_CENTER_WEIGHT_RANGE[1]
		);

		const gl = this._gl;
		const motionBlurProgram = this._programs.getMotionBlurProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(motionBlurProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);

		const uniforms = motionBlurProgram.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (uniforms.motionParams) {
			gl.uniform4f(
				uniforms.motionParams,
				shutterScale,
				maxSamples,
				velocityClamp,
				depthReject
			);
		}
		if (uniforms.centerWeight) {
			gl.uniform1f(uniforms.centerWeight, centerWeight);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyDOF(options?: DOFOptions): void {
		if (
			!this._sceneMotionTexture ||
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture ||
			!this._fullscreenVao
		) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const focusDistance = Math.max(
			0.01,
			finiteOr(options?.focusDistance, DEFAULT_DOF_OPTIONS.focusDistance)
		);
		const focusRange = Math.max(
			0.001,
			finiteOr(options?.focusRange, DEFAULT_DOF_OPTIONS.focusRange)
		);
		const nearStrength = sanitizeFiniteClamped(
			options?.nearStrength,
			DEFAULT_DOF_OPTIONS.nearStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const farStrength = sanitizeFiniteClamped(
			options?.farStrength,
			DEFAULT_DOF_OPTIONS.farStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const maxBlurRadius = sanitizeFiniteClamped(
			options?.maxBlurRadius,
			DEFAULT_DOF_OPTIONS.maxBlurRadius,
			DOF_MAX_BLUR_RADIUS_RANGE[0],
			DOF_MAX_BLUR_RADIUS_RANGE[1]
		);
		const depthCurve = sanitizeFiniteClamped(
			options?.depthCurve,
			DEFAULT_DOF_OPTIONS.depthCurve,
			DOF_DEPTH_CURVE_RANGE[0],
			DOF_DEPTH_CURVE_RANGE[1]
		);
		const highlightThreshold = Math.max(
			0,
			finiteOr(options?.highlightThreshold, DEFAULT_DOF_OPTIONS.highlightThreshold)
		);
		const highlightGain = sanitizeFiniteClamped(
			options?.highlightGain,
			DEFAULT_DOF_OPTIONS.highlightGain,
			DOF_HIGHLIGHT_GAIN_RANGE[0],
			DOF_HIGHLIGHT_GAIN_RANGE[1]
		);
		const chromaticAberration = sanitizeFiniteClamped(
			options?.chromaticAberration,
			DEFAULT_DOF_OPTIONS.chromaticAberration,
			DOF_CHROMATIC_ABERRATION_RANGE[0],
			DOF_CHROMATIC_ABERRATION_RANGE[1]
		);

		const gl = this._gl;
		const dofProgram = this._programs.getDOFProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(dofProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, this._sceneMotionTexture);

		const uniforms = dofProgram.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (uniforms.focusParams) {
			gl.uniform4f(
				uniforms.focusParams,
				focusDistance,
				focusRange,
				nearStrength,
				farStrength
			);
		}
		if (uniforms.dofParams) {
			gl.uniform4f(
				uniforms.dofParams,
				maxBlurRadius,
				depthCurve,
				highlightThreshold,
				highlightGain
			);
		}
		if (uniforms.chromaticAberration) {
			gl.uniform1f(uniforms.chromaticAberration, chromaticAberration);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyBloom(options?: BloomOptions): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture
		) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const bloomProgram = this._programs.getBloomProgram();
		const threshold = Math.max(
			0,
			finiteOr(options?.threshold, DEFAULT_BLOOM_OPTIONS.threshold)
		);
		const softKnee = Math.max(
			1e-4,
			finiteOr(options?.softKnee, DEFAULT_BLOOM_OPTIONS.softKnee)
		);
		const intensity = Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_BLOOM_OPTIONS.intensity)
		);
		const radius = clamp(
			finiteOr(options?.radius, DEFAULT_BLOOM_OPTIONS.radius),
			0.5,
			4
		);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._width, this._height);
		gl.useProgram(bloomProgram.program);
		gl.bindVertexArray(this._fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (bloomProgram.uniforms.sourceMap) {
			gl.uniform1i(bloomProgram.uniforms.sourceMap, 0);
		}
		if (bloomProgram.uniforms.texelSize) {
			gl.uniform2f(
				bloomProgram.uniforms.texelSize,
				1 / Math.max(1, this._width),
				1 / Math.max(1, this._height)
			);
		}
		if (bloomProgram.uniforms.bloomParams) {
			gl.uniform4f(
				bloomProgram.uniforms.bloomParams,
				threshold,
				softKnee,
				intensity,
				radius
			);
		}
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);

		this._presentSourceTexture = targetTexture;
	}

	private _applyFXAA(): void {
		if (
			!this._postFramebuffer ||
			!this._sceneColorTexture ||
			!this._postColorTexture
		) {
			return;
		}
		if (!this._fullscreenVao) {
			return;
		}
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}
		const targetTexture = this._resolvePostProcessTargetTexture(sourceTexture);
		if (!targetTexture) {
			return;
		}

		const gl = this._gl;
		const fxaaProgram = this._programs.getFXAAProgram();

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._postFramebuffer);
		this._bindPostSingleColorTarget(targetTexture);
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

		this._presentSourceTexture = targetTexture;
	}

	private _applyTAA(options?: TAAOptions): void {
		if (
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
		const sourceTexture = this._presentSourceTexture ?? this._sceneColorTexture;
		if (!sourceTexture) {
			return;
		}

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
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
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

		const weight = sanitizeFiniteClamped(
			options?.historyWeight,
			DEFAULT_TAA_OPTIONS.historyWeight,
			TAA_HISTORY_WEIGHT_RANGE[0],
			TAA_HISTORY_WEIGHT_RANGE[1]
		);
		const depthThreshold = sanitizeFiniteClamped(
			options?.disocclusionDepthThreshold,
			DEFAULT_TAA_OPTIONS.disocclusionDepthThreshold,
			TAA_DEPTH_THRESHOLD_RANGE[0],
			TAA_DEPTH_THRESHOLD_RANGE[1]
		);
		const motionFactor = sanitizeFiniteClamped(
			options?.motionFactor,
			DEFAULT_TAA_OPTIONS.motionFactor,
			TAA_MOTION_FACTOR_RANGE[0],
			TAA_MOTION_FACTOR_RANGE[1]
		);
		const varianceClampGamma = sanitizeFiniteClamped(
			options?.varianceClampGamma,
			DEFAULT_TAA_OPTIONS.varianceClampGamma,
			TAA_VARIANCE_GAMMA_RANGE[0],
			TAA_VARIANCE_GAMMA_RANGE[1]
		);
		const sharpen = sanitizeFiniteClamped(
			options?.sharpen,
			DEFAULT_TAA_OPTIONS.sharpen,
			TAA_SHARPEN_RANGE[0],
			TAA_SHARPEN_RANGE[1]
		);

		if (uniforms.historyWeight) gl.uniform1f(uniforms.historyWeight, weight);
		if (uniforms.depthThreshold)
			gl.uniform1f(uniforms.depthThreshold, depthThreshold);
		if (uniforms.motionFactor) gl.uniform1f(uniforms.motionFactor, motionFactor);
		if (uniforms.varianceClampGamma)
			gl.uniform1f(uniforms.varianceClampGamma, varianceClampGamma);
		if (uniforms.sharpen) gl.uniform1f(uniforms.sharpen, sharpen);
		if (uniforms.historyValid)
			gl.uniform1f(uniforms.historyValid, this._taaHistoryValid ? 1.0 : 0.0);

		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			null,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);

		this._taaHistoryIndex = 1 - historyIndex;
		this._taaHistoryValid = true;
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

	private _resolvePostProcessTargetTexture(
		sourceTexture: WebGLTexture
	): WebGLTexture | null {
		if (!this._sceneColorTexture || !this._postColorTexture) {
			return null;
		}
		if (sourceTexture === this._sceneColorTexture) {
			return this._postColorTexture;
		}
		if (sourceTexture === this._postColorTexture) {
			return this._sceneColorTexture;
		}
		return this._postColorTexture;
	}

	private _bindPostSingleColorTarget(texture: WebGLTexture): void {
		const gl = this._gl;
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			texture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			null,
			0
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
	}

	private _ensureFrameTargets(
		width: number,
		height: number,
		ssaoDownsample: number
	): void {
		if (
			this._sceneFramebuffer &&
			this._sceneColorTexture &&
			this._sceneMotionTexture &&
			this._sceneNormalTexture &&
			this._sceneDepthBuffer &&
			this._postFramebuffer &&
			this._postColorTexture &&
			this._ssaoRawTexture &&
			this._ssaoBlurTexture &&
			this._targetWidth === width &&
			this._targetHeight === height &&
			this._targetSSAODownsample === ssaoDownsample
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
		const normalInternalFormat = gl.RGBA8;
		const normalType = gl.UNSIGNED_BYTE;
		const aoWidth = Math.max(1, Math.floor(width / Math.max(ssaoDownsample, 1)));
		const aoHeight = Math.max(1, Math.floor(height / Math.max(ssaoDownsample, 1)));
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
		const sceneNormalTexture = this._createColorTexture(
			width,
			height,
			normalInternalFormat,
			normalType
		);
		const sceneDepthBuffer = gl.createRenderbuffer();
		const postFramebuffer = gl.createFramebuffer();
		const postColorTexture = this._createColorTexture(
			width,
			height,
			colorInternalFormat,
			colorType
		);
		const ssaoRawTexture = this._createColorTexture(
			aoWidth,
			aoHeight,
			colorInternalFormat,
			colorType
		);
		const ssaoBlurTexture = this._createColorTexture(
			aoWidth,
			aoHeight,
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
			if (sceneNormalTexture) gl.deleteTexture(sceneNormalTexture);
			if (sceneDepthBuffer) gl.deleteRenderbuffer(sceneDepthBuffer);
			if (postFramebuffer) gl.deleteFramebuffer(postFramebuffer);
			if (postColorTexture) gl.deleteTexture(postColorTexture);
			if (ssaoRawTexture) gl.deleteTexture(ssaoRawTexture);
			if (ssaoBlurTexture) gl.deleteTexture(ssaoBlurTexture);
			if (history0) gl.deleteTexture(history0);
			if (history1) gl.deleteTexture(history1);
			if (motionHistory0) gl.deleteTexture(motionHistory0);
			if (motionHistory1) gl.deleteTexture(motionHistory1);
		};

		if (
			!sceneFramebuffer ||
			!sceneColorTexture ||
			!sceneMotionTexture ||
			!sceneNormalTexture ||
			!sceneDepthBuffer ||
			!postFramebuffer ||
			!postColorTexture ||
			!ssaoRawTexture ||
			!ssaoBlurTexture ||
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
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT2,
			gl.TEXTURE_2D,
			sceneNormalTexture,
			0
		);
		gl.framebufferRenderbuffer(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.RENDERBUFFER,
			sceneDepthBuffer
		);
		gl.drawBuffers([
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		]);
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
		this._sceneNormalTexture = sceneNormalTexture;
		this._sceneDepthBuffer = sceneDepthBuffer;
		this._taaHistoryTextures = [history0, history1];
		this._taaMotionHistoryTextures = [motionHistory0, motionHistory1];
		this._taaHistoryIndex = 0;
		this._taaHistoryValid = false;
		this._postFramebuffer = postFramebuffer;
		this._postColorTexture = postColorTexture;
		this._ssaoRawTexture = ssaoRawTexture;
		this._ssaoBlurTexture = ssaoBlurTexture;
		this._presentSourceTexture = sceneColorTexture;
		this._targetWidth = width;
		this._targetHeight = height;
		this._targetSSAODownsample = ssaoDownsample;
		this._ssaoFrameIndex = 0;
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
		if (this._sceneNormalTexture) {
			gl.deleteTexture(this._sceneNormalTexture);
			this._sceneNormalTexture = null;
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
		this._taaHistoryValid = false;
		if (this._postFramebuffer) {
			gl.deleteFramebuffer(this._postFramebuffer);
			this._postFramebuffer = null;
		}
		if (this._postColorTexture) {
			gl.deleteTexture(this._postColorTexture);
			this._postColorTexture = null;
		}
		if (this._ssaoRawTexture) {
			gl.deleteTexture(this._ssaoRawTexture);
			this._ssaoRawTexture = null;
		}
		if (this._ssaoBlurTexture) {
			gl.deleteTexture(this._ssaoBlurTexture);
			this._ssaoBlurTexture = null;
		}
		this._presentSourceTexture = null;
		this._targetWidth = 0;
		this._targetHeight = 0;
		this._targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
		this._ssaoFrameIndex = 0;
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
		packed[offset + 1] = finiteOr(shadow.depthBias, 0);
		packed[offset + 2] = finiteOr(shadow.normalBias, 0);
		packed[offset + 3] = finiteOr(shadow.normalBiasMin, 0);
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
		packed[offset] = finiteOr(shadow.pcfRadius, 0);
		packed[offset + 1] = finiteOr(shadow.shadowStrength, 0);
		packed[offset + 2] = finiteOr(shadow.shadowMapSize, 0);
		packed[offset + 3] = finiteOr(shadow.atlasTileSize, 0);
	}
	return packed;
}

function flattenShadowParamsC(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = finiteOr(shadow.slopeBias, 0);
		packed[offset + 1] = 0;
		packed[offset + 2] = 0;
		packed[offset + 3] = 0;
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

function toFiniteColumnMajorMat4(matrix: Matrix4 | number[][]): Float32Array | null {
	const values = toColumnMajorMat4(matrix);
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return values;
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

function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function sanitizeFiniteClamped(
	value: unknown,
	fallback: number,
	minValue: number,
	maxValue: number
): number {
	return clamp(finiteOr(value, fallback), minValue, maxValue);
}

function sanitizeFloat32Array(
	values: Float32Array,
	fallback: number
): {
	values: Float32Array;
	hadInvalid: boolean;
} {
	let hadInvalid = false;
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			values[i] = fallback;
			hadInvalid = true;
		}
	}
	return { values, hadInvalid };
}

function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
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
