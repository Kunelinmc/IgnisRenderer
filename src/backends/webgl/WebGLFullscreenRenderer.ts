import type { FrameContext } from "../../pipeline/types";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import type { WebGLProgramLibrary } from "./WebGLProgramLibrary";

export interface WebGLFullscreenRendererHost {
	readonly gl: WebGL2RenderingContext;
	readonly targets: WebGLFrameTargetManager;
	getPrograms(): WebGLProgramLibrary;
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
export class WebGLFullscreenRenderer {
	private readonly _host: WebGLFullscreenRendererHost;
	public _vao: WebGLVertexArrayObject | null;

	public constructor(host: WebGLFullscreenRendererHost) {
		this._host = host;
		this._vao = host.gl.createVertexArray();
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
		const dirtyRects = this._host.resolveDirtyRects(
			context,
			viewportWidth,
			viewportHeight,
		);
		if (dirtyRects.length === 0) return;
		gl.enable(gl.SCISSOR_TEST);
		for (const rect of dirtyRects) {
			this._host.setScissorRect(
				rect.x,
				rect.y,
				rect.width,
				rect.height,
				viewportHeight,
			);
			gl.drawArrays(gl.TRIANGLES, 0, 3);
		}
		gl.disable(gl.SCISSOR_TEST);
	}

	public present(
		context: FrameContext | null,
		nonBlocking = false,
	): boolean {
		const source =
			this._host.targets._presentSourceTexture ??
			this._host.targets._sceneColorTexture;
		if (!source || !this._vao) return false;
		const programs = this._host.getPrograms();
		const program =
			nonBlocking ? programs.tryGetPresentProgram() : programs.getPresentProgram();
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
		if (!this._vao) return;
		this._host.gl.deleteVertexArray(this._vao);
		this._vao = null;
	}
}
