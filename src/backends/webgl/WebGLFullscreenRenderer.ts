import type { FrameContext } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
	WebGLProgramWarmupHandle,
} from "./WebGLProgramCompiler";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";

interface WebGLPresentProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLFullscreenRendererHost {
	readonly gl: WebGL2RenderingContext;
	readonly targets: WebGLFrameTargetManager;
	readonly programCompiler: WebGLProgramCompiler;
	getWidth(): number;
	getHeight(): number;
	isIncrementalPartial(context: FrameContext | null | undefined): boolean;
	resolveDirtyRects(
		context: FrameContext | null | undefined,
		width: number,
		height: number,
	): Array<{ x: number; y: number; width: number; height: number }>;
	setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number,
		viewportHeight: number,
	): void;
	markPresented(): void;
}

/** Owns the fullscreen VAO and all fullscreen WebGL presentation draws. */
export class WebGLFullscreenRenderer implements WebGLProgramWarmupContributor {
	private readonly _host: WebGLFullscreenRendererHost;
	private readonly _presentProgram: WebGLProgramSlot<WebGLPresentProgram>;
	public _vao: WebGLVertexArrayObject | null;

	constructor(host: WebGLFullscreenRendererHost) {
		this._host = host;
		this._vao = host.gl.createVertexArray();
		this._presentProgram = host.programCompiler.createSlot({
			label: "WebGLPresentProgram",
			vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
			fragment: () => ShaderSource.get("webgl.part.presentFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: {
					sourceMap: gl.getUniformLocation(program, "uSourceMap"),
				},
			}),
		});
	}

	public warmupPresentProgram(): WebGLProgramWarmupHandle {
		return this._presentProgram.warmup();
	}

	public collectWarmupTasks(
		_request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		return [
			{
				label: "WebGLPresentProgram",
				priority: "core",
				run: () => this.warmupPresentProgram(),
			},
		];
	}

	public draw(
		viewportWidth: number,
		viewportHeight: number,
		context: FrameContext | null | undefined,
	): void {
		const gl = this._host.gl;
		if (!this._host.isIncrementalPartial(context)) {
			gl.disable(gl.SCISSOR_TEST);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
			return;
		}
		const dirtyRects = this._host.resolveDirtyRects(context, viewportWidth, viewportHeight);
		if (dirtyRects.length === 0) return;
		gl.enable(gl.SCISSOR_TEST);
		for (const rect of dirtyRects) {
			this._host.setScissorRect(rect.x, rect.y, rect.width, rect.height, viewportHeight);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
		}
		gl.disable(gl.SCISSOR_TEST);
	}

	public present(context: FrameContext | null, nonBlocking = false): boolean {
		const source =
			this._host.targets._presentSourceTexture ?? this._host.targets._sceneColorTexture;
		if (!source || !this._vao) return false;
		const program = nonBlocking ? this._presentProgram.tryGet() : this._presentProgram.get();
		if (!program) return false;
		const gl = this._host.gl;
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, this._host.getWidth(), this._host.getHeight());
		gl.useProgram(program.program);
		gl.bindVertexArray(this._vao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, source);
		if (program.uniforms.sourceMap) gl.uniform1i(program.uniforms.sourceMap, 0);
		this.draw(this._host.getWidth(), this._host.getHeight(), context);
		gl.bindVertexArray(null);
		this._host.markPresented();
		return true;
	}

	public destroy(): void {
		this._presentProgram.destroy();
		if (this._vao) {
			this._host.gl.deleteVertexArray(this._vao);
			this._vao = null;
		}
	}
}
