import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import type { DrawPacket } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";

import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";
import type {
	WebGLAnimationPayloadPool,
	WebGLAnimationUniforms,
} from "./WebGLAnimationPayloadPool";
import type {
	WebGLGeometryHandle,
	WebGLSkinProfile,
} from "./WebGLGeometryRegistry";

export interface WebGLShadowDepthProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
	} & WebGLAnimationUniforms;
}

export interface WebGLShadowTransmittanceProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
		transmittance: WebGLUniformLocation | null;
	} & WebGLAnimationUniforms;
}

/** @internal A fully resolved atlas slice consumed by the WebGL shadow rasterizer. */
export interface WebGLShadowRasterSlice {
	readonly kind: "directional" | "spot";
	readonly lightIndex: number;
	readonly cascadeIndex: number;
	readonly viewportX: number;
	readonly viewportY: number;
	readonly viewportWidth: number;
	readonly viewportHeight: number;
	readonly viewProjectionMatrix: Matrix4;
}

/** @internal Runtime-owned, allocation-conscious input for one shadow raster frame. */
export interface WebGLShadowRasterPlan {
	readonly atlasTileSize: number;
	readonly atlasWidth: number;
	readonly atlasHeight: number;
	readonly slices: readonly WebGLShadowRasterSlice[];
	readonly sliceCount: number;
	readonly casterPackets: readonly DrawPacket[];
	readonly transmitterPackets: readonly DrawPacket[];
	readonly baselineFramebuffer: WebGLFramebuffer | null;
	readonly baselineViewportWidth: number;
	readonly baselineViewportHeight: number;
}

/** @internal Native target state returned to the owning shadow runtime. */
export interface WebGLShadowRasterPreparedState {
	readonly framebuffer: WebGLFramebuffer | null;
	readonly atlasTexture: WebGLTexture | null;
	readonly transmittanceTexture: WebGLTexture | null;
	readonly atlasTileSize: number;
	readonly depthProgramAvailable: boolean;
	readonly transmittanceProgramAvailable: boolean;
}

/** @internal Narrow device dependencies for the plan-only shadow rasterizer. */
export interface WebGLShadowRasterPassHost {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
	readonly geometry: {
		getGeometry(packet: DrawPacket): WebGLGeometryHandle | null;
	};
	readonly animationPayloads?: WebGLAnimationPayloadPool;
	readonly maxTextureSize: number;
}

type ShadowDeformationKey =
	| "static:0"
	| "static:1"
	| "skin4:0"
	| "skin4:1"
	| "skin8:0"
	| "skin8:1";

const SHADOW_DEFORMATION_KEYS: readonly ShadowDeformationKey[] = [
	"static:0",
	"static:1",
	"skin4:0",
	"skin4:1",
	"skin8:0",
	"skin8:1",
];

/** Executes prepared WebGL shadow raster plans and owns their native atlas targets. */
export class WebGLShadowRasterPass {
	private readonly _host: WebGLShadowRasterPassHost;
	private readonly _depthProgramSlots = new Map<
		ShadowDeformationKey,
		WebGLProgramSlot<WebGLShadowDepthProgram>
	>();
	private readonly _transmittanceProgramSlots = new Map<
		ShadowDeformationKey,
		WebGLProgramSlot<WebGLShadowTransmittanceProgram>
	>();
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowTransmittanceTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();

	constructor(host: WebGLShadowRasterPassHost) {
		this._host = host;
		for (const key of SHADOW_DEFORMATION_KEYS) {
			this._depthProgramSlots.set(key, host.programCompiler.createSlot({
				label: key === "static:0" ?
					"WebGLShadowDepthProgram" : `WebGLShadowDepthProgram_${key}`,
				vertex: () => getShadowSource("depth", key, "vertex"),
				fragment: () => getShadowSource("depth", key, "fragment"),
				reflect: (gl, program) => ({
					program,
					uniforms: {
						mvp: gl.getUniformLocation(program, "uMvp"),
						...createAnimationUniforms(gl, program),
					},
				}),
			}));
			this._transmittanceProgramSlots.set(key, host.programCompiler.createSlot({
				label: key === "static:0" ?
					"WebGLShadowTransmittanceProgram" :
					`WebGLShadowTransmittanceProgram_${key}`,
				vertex: () => getShadowSource("transmittance", key, "vertex"),
				fragment: () => getShadowSource("transmittance", key, "fragment"),
				reflect: (gl, program) => ({
					program,
					uniforms: {
						mvp: gl.getUniformLocation(program, "uMvp"),
						transmittance: gl.getUniformLocation(program, "uTransmittance"),
						...createAnimationUniforms(gl, program),
					},
				}),
			}));
		}
	}

	public warmupPrograms(): WebGLProgramWarmupHandle[] {
		return [
			...Array.from(this._depthProgramSlots.values(), (slot) => slot.warmup()),
			...Array.from(this._transmittanceProgramSlots.values(), (slot) => slot.warmup()),
		];
	}

	/** Prepares native targets and shader programs without inspecting frame state. */
	public prepare(plan: WebGLShadowRasterPlan): WebGLShadowRasterPreparedState {
		try {
			this._ensureShadowTargets(plan);
			let depthProgramAvailable = !!this._resolveDepthProgram("static:0");
			for (const packet of plan.casterPackets) {
				const geometry = this._host.geometry.getGeometry(packet);
				if (geometry && !this._resolveDepthProgram(profileKey(geometry))) {
					depthProgramAvailable = false;
				}
			}
			let transmittanceProgramAvailable =
				!!this._resolveTransmittanceProgram("static:0");
			for (const packet of plan.transmitterPackets) {
				const geometry = this._host.geometry.getGeometry(packet);
				if (geometry && !this._resolveTransmittanceProgram(profileKey(geometry))) {
					transmittanceProgramAvailable = false;
				}
			}
			return {
				framebuffer: this._shadowFramebuffer,
				atlasTexture: this._shadowAtlasTexture,
				transmittanceTexture: this._shadowTransmittanceTexture,
				atlasTileSize: this._shadowAtlasTileSize,
				depthProgramAvailable,
				transmittanceProgramAvailable,
			};
		} finally {
			this._restoreFrameBaseline(plan);
		}
	}

	/** Renders one previously prepared plan and restores the frame baseline. */
	public render(plan: WebGLShadowRasterPlan): void {
		if (
			!this._shadowFramebuffer ||
			!this._shadowAtlasTexture ||
			plan.atlasTileSize !== this._shadowAtlasTileSize
		)
			return;

		const gl = this._host.gl;
		try {
			gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFramebuffer);
			gl.disable(gl.BLEND);
			gl.enable(gl.DEPTH_TEST);
			gl.depthMask(true);
			gl.colorMask(false, false, false, false);
			gl.enable(gl.SCISSOR_TEST);
			gl.clearDepth(1);

			for (let index = 0; index < plan.sliceCount; index++) {
				this._renderDepthSlice(plan, plan.slices[index]);
			}

			gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
			gl.colorMask(true, true, true, true);
			gl.depthMask(false);
			gl.enable(gl.BLEND);
			gl.blendFuncSeparate(gl.ZERO, gl.SRC_COLOR, gl.ZERO, gl.ONE);
			gl.clearColor(1, 1, 1, 1);
			for (let index = 0; index < plan.sliceCount; index++) {
				this._renderTransmittanceSlice(
					plan,
					plan.slices[index],
				);
			}
		} finally {
			this._restoreFrameBaseline(plan);
		}
	}

	public destroy(): void {
		for (const slot of this._depthProgramSlots.values()) slot.destroy();
		for (const slot of this._transmittanceProgramSlots.values()) slot.destroy();
		this._depthProgramSlots.clear();
		this._transmittanceProgramSlots.clear();
		this._destroyShadowTargets();
	}

	private _renderDepthSlice(
		plan: WebGLShadowRasterPlan,
		slice: WebGLShadowRasterSlice | undefined,
	): void {
		if (!slice) return;
		this._setSliceViewport(slice);
		this._host.gl.clear(this._host.gl.DEPTH_BUFFER_BIT);
		for (const packet of plan.casterPackets) {
			this._drawShadowPacket(packet, slice.viewProjectionMatrix);
		}
	}

	private _renderTransmittanceSlice(
		plan: WebGLShadowRasterPlan,
		slice: WebGLShadowRasterSlice | undefined,
	): void {
		if (!slice) return;
		this._setSliceViewport(slice);
		this._host.gl.clear(this._host.gl.COLOR_BUFFER_BIT);
		if (plan.transmitterPackets.length === 0) return;
		for (const packet of plan.transmitterPackets) {
			this._drawShadowTransmittancePacket(packet, slice.viewProjectionMatrix);
		}
	}

	private _setSliceViewport(slice: WebGLShadowRasterSlice): void {
		const gl = this._host.gl;
		gl.viewport(slice.viewportX, slice.viewportY, slice.viewportWidth, slice.viewportHeight);
		gl.scissor(slice.viewportX, slice.viewportY, slice.viewportWidth, slice.viewportHeight);
	}

	private _drawShadowPacket(
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4,
	): void {
		if (!Matrix4.isFinite(packet.submission.instance.worldMatrix)) return;
		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._host.gl.TRIANGLES) return;
		const shadowProgram = this._resolveDepthProgram(profileKey(geometry));
		if (!shadowProgram) return;

		Matrix4.multiply(viewProjectionMatrix, packet.submission.instance.worldMatrix, this._shadowMvpMatrix);
		const gl = this._host.gl;
		gl.useProgram(shadowProgram.program);
		if (
			this._host.animationPayloads &&
			!this._host.animationPayloads.bind(shadowProgram.uniforms, packet, geometry)
		) {
			return;
		}
		if (shadowProgram.uniforms.mvp) {
			gl.uniformMatrix4fv(
				shadowProgram.uniforms.mvp,
				false,
				Matrix4.toColumnMajorArray(this._shadowMvpMatrix),
			);
		}
		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(geometry.topology, geometry.indexCount, geometry.indexType, 0);
		gl.bindVertexArray(null);
	}

	private _drawShadowTransmittancePacket(
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4,
	): void {
		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._host.gl.TRIANGLES) return;
		const shadowProgram = this._resolveTransmittanceProgram(profileKey(geometry));
		if (!shadowProgram) return;
		if (
			!Matrix4.isFinite(packet.submission.instance.worldMatrix) ||
			!Matrix4.isFinite(viewProjectionMatrix)
		) {
			return;
		}
		const gl = this._host.gl;
		gl.useProgram(shadowProgram.program);
		if (
			this._host.animationPayloads &&
			!this._host.animationPayloads.bind(shadowProgram.uniforms, packet, geometry)
		) {
			return;
		}
		Matrix4.multiply(viewProjectionMatrix, packet.submission.instance.worldMatrix, this._shadowMvpMatrix);
		gl.uniformMatrix4fv(
			shadowProgram.uniforms.mvp,
			false,
			Matrix4.toColumnMajorArray(this._shadowMvpMatrix),
		);
		const transmittance = resolveMaterialShadowTransmittance(packet.submission.material.effective);
		gl.uniform3f(
			shadowProgram.uniforms.transmittance,
			transmittance.r,
			transmittance.g,
			transmittance.b,
		);
		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(geometry.topology, geometry.indexCount, geometry.indexType, 0);
		gl.bindVertexArray(null);
	}

	private _resolveDepthProgram(key: ShadowDeformationKey): WebGLShadowDepthProgram | null {
		return this._depthProgramSlots.get(key)?.tryGet() ?? null;
	}

	private _resolveTransmittanceProgram(
		key: ShadowDeformationKey,
	): WebGLShadowTransmittanceProgram | null {
		return this._transmittanceProgramSlots.get(key)?.tryGet() ?? null;
	}

	private _ensureShadowTargets(plan: WebGLShadowRasterPlan): void {
		const tileSize = plan.atlasTileSize;
		if (
			this._shadowFramebuffer &&
			this._shadowAtlasTexture &&
			this._shadowTransmittanceTexture &&
			this._shadowAtlasTileSize === tileSize
		)
			return;
		const atlasWidth = plan.atlasWidth;
		const atlasHeight = plan.atlasHeight;
		if (atlasWidth > this._host.maxTextureSize || atlasHeight > this._host.maxTextureSize) {
			throw new Error(
				`WebGL shadow atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._host.maxTextureSize}`,
			);
		}

		const gl = this._host.gl;
		const shadowTexture = gl.createTexture();
		const transmittanceTexture = gl.createTexture();
		const shadowFramebuffer = gl.createFramebuffer();
		if (!shadowTexture || !transmittanceTexture || !shadowFramebuffer) {
			if (shadowTexture) gl.deleteTexture(shadowTexture);
			if (transmittanceTexture) gl.deleteTexture(transmittanceTexture);
			if (shadowFramebuffer) gl.deleteFramebuffer(shadowFramebuffer);
			throw new Error("Failed to create WebGL shadow atlas targets");
		}

		try {
			this._allocateDepthAtlas(shadowTexture, atlasWidth, atlasHeight);
			this._allocateTransmittanceAtlas(transmittanceTexture, atlasWidth, atlasHeight);
			gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.DEPTH_ATTACHMENT,
				gl.TEXTURE_2D,
				shadowTexture,
				0,
			);
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.COLOR_ATTACHMENT0,
				gl.TEXTURE_2D,
				transmittanceTexture,
				0,
			);
			gl.drawBuffers([gl.NONE]);
			gl.readBuffer(gl.NONE);
			const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
			if (status !== gl.FRAMEBUFFER_COMPLETE) {
				throw new Error(
					`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`,
				);
			}
		} catch (error) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			gl.deleteTexture(transmittanceTexture);
			throw error;
		}

		this._destroyShadowTargets();
		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowTransmittanceTexture = transmittanceTexture;
		this._shadowAtlasTileSize = tileSize;
	}

	private _allocateDepthAtlas(texture: WebGLTexture, width: number, height: number): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		this._setAtlasTextureParameters();
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.DEPTH_COMPONENT24,
			width,
			height,
			0,
			gl.DEPTH_COMPONENT,
			gl.UNSIGNED_INT,
			null,
		);
	}

	private _allocateTransmittanceAtlas(
		texture: WebGLTexture,
		width: number,
		height: number,
	): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		this._setAtlasTextureParameters();
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA8,
			width,
			height,
			0,
			gl.RGBA,
			gl.UNSIGNED_BYTE,
			null,
		);
	}

	private _setAtlasTextureParameters(): void {
		const gl = this._host.gl;
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	private _destroyShadowTargets(): void {
		const gl = this._host.gl;
		if (this._shadowFramebuffer) gl.deleteFramebuffer(this._shadowFramebuffer);
		if (this._shadowAtlasTexture) gl.deleteTexture(this._shadowAtlasTexture);
		if (this._shadowTransmittanceTexture) {
			gl.deleteTexture(this._shadowTransmittanceTexture);
		}
		this._shadowFramebuffer = null;
		this._shadowAtlasTexture = null;
		this._shadowTransmittanceTexture = null;
		this._shadowAtlasTileSize = 0;
	}

	private _restoreFrameBaseline(plan: WebGLShadowRasterPlan): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, null);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.depthMask(true);
		gl.disable(gl.SCISSOR_TEST);
		gl.colorMask(true, true, true, true);
		gl.bindFramebuffer(gl.FRAMEBUFFER, plan.baselineFramebuffer);
		gl.drawBuffers([plan.baselineFramebuffer ? gl.COLOR_ATTACHMENT0 : gl.BACK]);
		gl.bindVertexArray(null);
		gl.bindFramebuffer(gl.FRAMEBUFFER, plan.baselineFramebuffer);
		gl.viewport(0, 0, plan.baselineViewportWidth, plan.baselineViewportHeight);
	}
}

function profileKey(geometry: WebGLGeometryHandle): ShadowDeformationKey {
	const skinProfile =
		geometry.skinProfile === "skin4" || geometry.skinProfile === "skin8" ?
			geometry.skinProfile : "static";
	return `${skinProfile}:${geometry.morphPositionTexture ? 1 : 0}` as
		ShadowDeformationKey;
}

function getShadowSource(
	kind: "depth" | "transmittance",
	key: ShadowDeformationKey,
	stage: "vertex" | "fragment",
): string {
	const [skinProfile, morphPosition] = key.split(":") as [WebGLSkinProfile, string];
	const artifact = ShaderSource.get(`webgl.shadow.${kind}`, {
		specialization: {
			skinProfile,
			morphPosition: morphPosition === "1",
		},
	});
	return artifact.stages[stage]!.code;
}

function createAnimationUniforms(
	gl: WebGL2RenderingContext,
	program: WebGLProgram,
): WebGLAnimationUniforms {
	return {
		animationPayload: gl.getUniformLocation(program, "uAnimationPayload"),
		morphPositionDeltas: gl.getUniformLocation(
			program,
			"uMorphPositionDeltas",
		),
		morphNormalDeltas: gl.getUniformLocation(program, "uMorphNormalDeltas"),
		animationCounts: gl.getUniformLocation(program, "uAnimationCounts"),
		animationOffsets: gl.getUniformLocation(program, "uAnimationOffsets"),
		animationTextureWidths: gl.getUniformLocation(
			program,
			"uAnimationTextureWidths",
		),
	};
}
