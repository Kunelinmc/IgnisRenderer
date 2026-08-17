import { ParticleBlendMode } from "../../particles";
import { materialUsesTransmission } from "../../materials/transparency";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";
import type { WebGLParticleRenderOptions } from "./WebGLParticlePass";
import type { WebGLSceneRenderOptions } from "./WebGLScenePass";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";
import { Logger } from "../../foundation/Logger";

interface WebGLCopyProgram {
	program: WebGLProgram;
	uniforms: { sourceMap: WebGLUniformLocation | null };
}

interface WebGLOITResolveProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		oitAccumMap: WebGLUniformLocation | null;
		oitRevealMap: WebGLUniformLocation | null;
	};
}

export interface WebGLTransparencyRuntimeHost {
	readonly gl: WebGL2RenderingContext;
	readonly targets: WebGLFrameTargetManager;
	readonly programCompiler: WebGLProgramCompiler;
	getFullscreenVao(): WebGLVertexArrayObject | null;
	getWidth(): number;
	getHeight(): number;
	renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean,
		options?: WebGLSceneRenderOptions,
	): void;
	renderParticles(context: FrameContext, options?: WebGLParticleRenderOptions): void;
	drawFullscreen(width: number, height: number, context: FrameContext | null): void;
}

/** Owns WebGL OIT routing and all per-frame transparency state. */
export class WebGLTransparencyRuntime {
	private readonly _host: WebGLTransparencyRuntimeHost;
	private readonly _copyProgram: WebGLProgramSlot<WebGLCopyProgram>;
	private readonly _oitResolveProgram: WebGLProgramSlot<WebGLOITResolveProgram>;
	private _active = false;
	private _hasContributors = false;
	private _sceneColorCopyReady = false;
	private _transparentPackets: DrawPacket[] = [];
	private _legacyPackets: DrawPacket[] = [];

	constructor(host: WebGLTransparencyRuntimeHost) {
		this._host = host;
		this._copyProgram = host.programCompiler.createSlot({
			label: "WebGLCopyProgram",
			vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
			fragment: () => ShaderSource.get("webgl.part.copyFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: { sourceMap: gl.getUniformLocation(program, "uSourceMap") },
			}),
		});
		this._oitResolveProgram = host.programCompiler.createSlot({
			label: "WebGLOITResolveProgram",
			vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
			fragment: () => ShaderSource.get("webgl.part.oitResolveFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: {
					sceneColor: gl.getUniformLocation(program, "uSceneColor"),
					oitAccumMap: gl.getUniformLocation(program, "uOITAccumMap"),
					oitRevealMap: gl.getUniformLocation(program, "uOITRevealMap"),
				},
			}),
		});
	}

	public warmupCopyProgram(): WebGLProgramWarmupHandle {
		return this._copyProgram.warmup();
	}

	public warmupOITResolveProgram(): WebGLProgramWarmupHandle {
		return this._oitResolveProgram.warmup();
	}

	public destroy(): void {
		this._copyProgram.destroy();
		this._oitResolveProgram.destroy();
	}

	public beginFrame(context: FrameContext): void {
		this._configure(context);
		this._hasContributors = false;
		this._sceneColorCopyReady = false;
		this._transparentPackets = [];
		this._legacyPackets = [];
	}

	public abortFrame(): void {
		this._active = false;
		this._hasContributors = false;
		this._sceneColorCopyReady = false;
		this._transparentPackets = [];
		this._legacyPackets = [];
	}

	public isActive(): boolean {
		return this._active && !!this._host.targets._oitFramebuffer;
	}

	public prepareTransparent(context: FrameContext): void {
		if (!this.isActive()) {
			this._host.renderPackets(context, context.scene.transparentPackets, true);
			return;
		}
		const partition = this._partition(context.scene.transparentPackets);
		this._transparentPackets = partition.oitPackets;
		this._legacyPackets = partition.legacyPackets;
		this._hasContributors = false;
		if (this._transparentPackets.length > 0) this._clearTargets();
	}

	public renderTransparentAccum(context: FrameContext): void {
		if (!this.isActive() || this._transparentPackets.length === 0) return;
		this._host.renderPackets(context, this._transparentPackets, true, {
			framebuffer: this._host.targets._oitFramebuffer,
			drawBuffers: [this._host.gl.COLOR_ATTACHMENT0],
			blendMode: "oit-accum",
			oitPassMode: 1,
		});
	}

	public renderTransparentReveal(context: FrameContext): void {
		if (!this.isActive() || this._transparentPackets.length === 0) return;
		this._host.renderPackets(context, this._transparentPackets, true, {
			framebuffer: this._host.targets._oitFramebuffer,
			drawBuffers: [this._host.gl.COLOR_ATTACHMENT0],
			blendMode: "oit-reveal",
			oitPassMode: 2,
		});
		this._hasContributors = true;
	}

	public prepareParticles(): void {
		if (this.isActive() && !this._hasContributors) this._clearTargets();
	}

	public renderParticleAccum(context: FrameContext): void {
		if (!this.isActive()) {
			this._host.renderParticles(context);
			return;
		}
		this._host.renderParticles(context, {
			framebuffer: this._host.targets._oitFramebuffer,
			drawBuffers: [this._host.gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 1,
		});
	}

	public renderParticleReveal(context: FrameContext): void {
		if (!this.isActive()) return;
		this._host.renderParticles(context, {
			framebuffer: this._host.targets._oitFramebuffer,
			drawBuffers: [this._host.gl.COLOR_ATTACHMENT0],
			includeBlendModes: [ParticleBlendMode.Alpha],
			oitPassMode: 2,
		});
		this._hasContributors = true;
	}

	public copySceneColor(context: FrameContext): void {
		this._sceneColorCopyReady = this._hasContributors && this._copySceneColor(context);
	}

	public resolve(context: FrameContext): void {
		if (this._hasContributors && this._sceneColorCopyReady) {
			this._resolveComposition(context);
		}
	}

	public renderLegacy(context: FrameContext): void {
		if (this._legacyPackets.length > 0) {
			this._host.renderPackets(context, this._legacyPackets, true);
		}
		this._transparentPackets = [];
		this._legacyPackets = [];
		this._hasContributors = false;
		this._sceneColorCopyReady = false;
	}

	public renderLegacyTransparent(context: FrameContext): void {
		if (
			context.scene.transparentPackets.some((packet) =>
				materialUsesTransmission(packet.material),
			)
		) {
			this._copyOpaqueLinearDepth(context);
		}
		for (const packet of context.scene.transparentPackets) {
			if (materialUsesTransmission(packet.material)) {
				if (this._copyTransmissionBackground(context)) {
					const gl = this._host.gl;
					gl.bindTexture(
						gl.TEXTURE_2D,
						this._host.targets._transmissionBackgroundTexture,
					);
					gl.generateMipmap(gl.TEXTURE_2D);
				}
				this._host.renderPackets(context, [packet], true, {
					blendMode: "disabled",
				});
			} else {
				this._host.renderPackets(context, [packet], true);
			}
		}
	}

	public prepareTransmissionDepth(context: FrameContext): void {
		this._copyOpaqueLinearDepth(context);
	}

	public renderLegacyTransparentSegment(context: FrameContext, start: number, end: number): void {
		const packets = context.scene.transparentPackets.slice(start, end);
		if (packets.length > 0) this._host.renderPackets(context, packets, true);
	}

	public copyTransmissionBackground(context: FrameContext): void {
		if (!this._copyTransmissionBackground(context)) return;
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, this._host.targets._transmissionBackgroundTexture);
		gl.generateMipmap(gl.TEXTURE_2D);
	}

	public renderTransmissionPacket(context: FrameContext, index: number): void {
		const packet = context.scene.transparentPackets[index];
		if (!packet || !materialUsesTransmission(packet.material)) return;
		this._host.renderPackets(context, [packet], true, {
			blendMode: "disabled",
		});
	}

	public renderParticlesLegacy(context: FrameContext): void {
		this._host.renderParticles(context);
	}

	public renderAdditiveParticles(context: FrameContext): void {
		this._host.renderParticles(context, {
			includeBlendModes: [ParticleBlendMode.Additive],
		});
	}

	private _configure(context: FrameContext): void {
		if (
			(context.scene?.transparentPackets ?? []).some((packet) =>
				materialUsesTransmission(packet.material),
			)
		) {
			this._active = false;
			return;
		}
		if (context.features.enableOIT !== true) {
			this._active = false;
			return;
		}
		const targets = this._host.targets;
		if (
			!targets._oitFramebuffer ||
			!targets._oitAccumTexture ||
			!targets._oitRevealTexture ||
			!targets._postFramebuffer ||
			!targets._postColorTexture
		) {
			const key = "webgl-oit-disabled-runtime";
			Logger.warn(
				`[${key}] WebGL OIT requires float color-buffer render targets; falling back to legacy transparent rendering.`,
				{ scope: "WebGLTransparencyRuntime", onceKey: key },
			);
			this._active = false;
			return;
		}
		this._active = true;
	}

	private _partition(packets: DrawPacket[]): {
		oitPackets: DrawPacket[];
		legacyPackets: DrawPacket[];
	} {
		const oitPackets: DrawPacket[] = [];
		const legacyPackets: DrawPacket[] = [];
		for (const packet of packets) {
			if (
				materialUsesTransmission(packet.material) ||
				packet.material instanceof ShaderMaterial
			) {
				legacyPackets.push(packet);
			} else {
				oitPackets.push(packet);
			}
		}
		return { oitPackets, legacyPackets };
	}

	private _clearTargets(): void {
		const targets = this._host.targets;
		if (!targets._oitFramebuffer || !targets._oitAccumTexture || !targets._oitRevealTexture) {
			return;
		}
		const gl = this._host.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, targets._oitFramebuffer);
		targets.bindOITSingleColorTarget(targets._oitAccumTexture);
		gl.viewport(0, 0, this._host.getWidth(), this._host.getHeight());
		gl.disable(gl.BLEND);
		gl.clearColor(0, 0, 0, 0);
		gl.clear(gl.COLOR_BUFFER_BIT);
		targets.bindOITSingleColorTarget(targets._oitRevealTexture);
		gl.clearColor(1, 1, 1, 1);
		gl.clear(gl.COLOR_BUFFER_BIT);
	}

	private _copySceneColor(context: FrameContext): boolean {
		return this._copyTextureTo(
			context,
			this._host.targets._sceneColorTexture,
			this._host.targets._postColorTexture,
		);
	}

	private _copyTransmissionBackground(context: FrameContext): boolean {
		return this._copyTextureTo(
			context,
			this._host.targets._sceneColorTexture,
			this._host.targets._transmissionBackgroundTexture,
		);
	}

	private _copyOpaqueLinearDepth(context: FrameContext): boolean {
		return this._copyTextureTo(
			context,
			this._host.targets._sceneMotionTexture,
			this._host.targets._transmissionDepthTexture,
		);
	}

	private _copyTextureTo(
		context: FrameContext,
		sourceTexture: WebGLTexture | null,
		targetTexture: WebGLTexture | null,
	): boolean {
		const targets = this._host.targets;
		if (
			!targets._postFramebuffer ||
			!targetTexture ||
			!sourceTexture ||
			!this._host.getFullscreenVao()
		)
			return false;
		const program = this._copyProgram.get();
		const gl = this._host.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, targets._postFramebuffer);
		targets.bindPostSingleColorTarget(targetTexture);
		gl.viewport(0, 0, this._host.getWidth(), this._host.getHeight());
		gl.useProgram(program.program);
		gl.bindVertexArray(this._host.getFullscreenVao());
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (program.uniforms.sourceMap) gl.uniform1i(program.uniforms.sourceMap, 0);
		// Copies must cover preserved regions in an incremental frame as well.
		this._host.drawFullscreen(this._host.getWidth(), this._host.getHeight(), null);
		gl.bindVertexArray(null);
		return true;
	}

	private _resolveComposition(context: FrameContext): void {
		const targets = this._host.targets;
		if (
			!this._hasContributors ||
			!targets._sceneFramebuffer ||
			!targets._sceneColorTexture ||
			!targets._postColorTexture ||
			!targets._oitAccumTexture ||
			!targets._oitRevealTexture ||
			!this._host.getFullscreenVao()
		)
			return;
		const program = this._oitResolveProgram.tryGet();
		if (!program) return;
		const gl = this._host.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, targets._sceneFramebuffer);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.viewport(0, 0, this._host.getWidth(), this._host.getHeight());
		gl.useProgram(program.program);
		gl.bindVertexArray(this._host.getFullscreenVao());
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		const textures = [
			[gl.TEXTURE0, targets._postColorTexture, program.uniforms.sceneColor],
			[gl.TEXTURE1, targets._oitAccumTexture, program.uniforms.oitAccumMap],
			[gl.TEXTURE2, targets._oitRevealTexture, program.uniforms.oitRevealMap],
		] as const;
		for (let index = 0; index < textures.length; index++) {
			const [unit, texture, uniform] = textures[index];
			gl.activeTexture(unit);
			gl.bindTexture(gl.TEXTURE_2D, texture);
			if (uniform) gl.uniform1i(uniform, index);
		}
		this._host.drawFullscreen(this._host.getWidth(), this._host.getHeight(), context);
		gl.bindVertexArray(null);
		gl.activeTexture(gl.TEXTURE0);
		targets._presentSourceTexture = targets._sceneColorTexture;
	}
}

/** @internal Adapts transparency program ownership into warmup tasks. */
export class WebGLTransparencyWarmupContributor
	implements WebGLProgramWarmupContributor
{
	private readonly _runtime: WebGLTransparencyRuntime;

	public constructor(runtime: WebGLTransparencyRuntime) {
		this._runtime = runtime;
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		const tasks: WebGLProgramWarmupTask[] = [];
		if (request.plan.materials.some((material) => materialUsesTransmission(material))) {
			tasks.push({
				label: "WebGLCopyProgram",
				priority: "core",
				run: () => this._runtime.warmupCopyProgram(),
			});
		}
		if (request.context.features?.enableOIT) {
			tasks.push({
				label: "WebGLOITResolveProgram",
				priority: "optional",
				run: () => this._runtime.warmupOITResolveProgram(),
			});
		}
		return tasks;
	}
}
