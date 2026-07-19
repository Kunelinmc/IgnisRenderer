import { ParticleBlendMode } from "../../particles";
import { materialUsesTransmission } from "../../materials/transparency";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type { WebGLProgramLibrary } from "./WebGLProgramLibrary";
import type { WebGLParticleRenderOptions } from "./WebGLParticlePass";
import type { WebGLSceneRenderOptions } from "./WebGLScenePass";
import { Logger } from "../../foundation/Logger";

export interface WebGLTransparencyRuntimeHost {
	readonly gl: WebGL2RenderingContext;
	readonly targets: WebGLFrameTargetManager;
	getPrograms(): WebGLProgramLibrary;
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
	drawFullscreen(width: number, height: number, context: FrameContext): void;
}

/** Owns WebGL OIT routing and all per-frame transparency state. */
export class WebGLTransparencyRuntime {
	private readonly _host: WebGLTransparencyRuntimeHost;
	private _active = false;
	private _hasContributors = false;
	private _transparentPackets: DrawPacket[] = [];
	private _legacyPackets: DrawPacket[] = [];

	public constructor(host: WebGLTransparencyRuntimeHost) {
		this._host = host;
	}

	public beginFrame(context: FrameContext): void {
		this._configure(context);
		this._hasContributors = false;
		this._transparentPackets = [];
		this._legacyPackets = [];
	}

	public abortFrame(): void {
		this._active = false;
		this._hasContributors = false;
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

	public resolve(context: FrameContext): void {
		if (this._hasContributors) this._resolveComposition(context);
	}

	public renderLegacy(context: FrameContext): void {
		if (this._legacyPackets.length > 0) {
			this._host.renderPackets(context, this._legacyPackets, true);
		}
		this._transparentPackets = [];
		this._legacyPackets = [];
		this._hasContributors = false;
	}

	public renderLegacyTransparent(context: FrameContext): void {
		this._host.renderPackets(context, context.scene.transparentPackets, true);
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
		const targets = this._host.targets;
		if (
			!targets._postFramebuffer ||
			!targets._postColorTexture ||
			!targets._sceneColorTexture ||
			!this._host.getFullscreenVao()
		) return false;
		const program = this._host.getPrograms().tryGetCopyProgram();
		if (!program) return false;
		const gl = this._host.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, targets._postFramebuffer);
		targets.bindPostSingleColorTarget(targets._postColorTexture);
		gl.viewport(0, 0, this._host.getWidth(), this._host.getHeight());
		gl.useProgram(program.program);
		gl.bindVertexArray(this._host.getFullscreenVao());
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, targets._sceneColorTexture);
		if (program.uniforms.sourceMap) gl.uniform1i(program.uniforms.sourceMap, 0);
		this._host.drawFullscreen(this._host.getWidth(), this._host.getHeight(), context);
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
			!this._host.getFullscreenVao() ||
			!this._copySceneColor(context)
		) return;
		const program = this._host.getPrograms().tryGetOITResolveProgram();
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
