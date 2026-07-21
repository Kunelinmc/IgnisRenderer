import type { FrameContext } from "../../pipeline/types";
import type { IncrementalDirtyRect } from "../../pipeline/incremental";
import type { ICommandEncoder } from "../../backends/ICommandEncoder";
import type { IRenderTexture } from "../../backends/types";
import type { WebGPUPostProcessFrameTargets } from "../../backends/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../backends/webgpu/postprocess/PostProcessSharedContext";
import type { WebGLProgramCompiler } from "../../backends/webgl/WebGLProgramCompiler";
import type { PostProcessResourceAccessor } from "../types";
export type { IncrementalDirtyRect } from "../../pipeline/incremental";

export type EmptyOptions = Record<string, never>;

/** @internal Software context supplied to built-in screen post-process implementations. */
export interface SoftwareBuiltinPostProcessContext {
	readonly canvasContext: CanvasRenderingContext2D | null;
	readonly resources: PostProcessResourceAccessor<ArrayBufferView>;
}

/** @internal WebGPU context supplied to built-in screen post-process implementations. */
export interface WebGPUScreenPostProcessContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly resources: PostProcessResourceAccessor<IRenderTexture>;
	readonly frameBinding?: unknown;
	readonly lightingState?: unknown;
	getFrameData<T>(key: unknown): T | undefined;
}

/** @internal WebGPU runtime context supplied to built-in screen implementations. */
export type WebGPURuntimePostProcessContext = WebGPUScreenPostProcessContext;

/** @internal WebGPU context supplied to the built-in gamma implementation. */
export type WebGPUGammaContext = WebGPUScreenPostProcessContext;

/** @internal WebGL context supplied to built-in screen post-process implementations. */
export interface WebGLScreenPostProcessContext {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	readonly resources: PostProcessResourceAccessor<WebGLTexture>;
	getSourceTexture(): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(): void;
	warn(key: string, message: string): void;
}

export function resolveSoftwareDirtyRects(context: FrameContext): IncrementalDirtyRect[] {
	const width = Math.max(1, context.attachments.width);
	const height = Math.max(1, context.attachments.height);
	const incremental = context.incremental;
	if (
		!incremental.enabled ||
		incremental.forceFullFrame ||
		incremental.dirtyRects.length === 0
	) {
		return [{ minX: 0, minY: 0, maxX: width - 1, maxY: height - 1 }];
	}
	const dirtyRects: IncrementalDirtyRect[] = [];
	for (const rect of incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
		const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
		if (minX > maxX || minY > maxY) {
			continue;
		}
		dirtyRects.push({ minX, minY, maxX, maxY });
	}
	return dirtyRects;
}

export function forEachSoftwareDirtyRect(
	dirtyRects: IncrementalDirtyRect[],
	callback: (rect: IncrementalDirtyRect) => void
): void {
	for (const rect of dirtyRects) {
		if (rect.minX > rect.maxX || rect.minY > rect.maxY) {
			continue;
		}
		callback(rect);
	}
}

export function softwareRectIntersectsDirtyRects(
	minX: number,
	minY: number,
	maxX: number,
	maxY: number,
	dirtyRects: IncrementalDirtyRect[]
): boolean {
	for (const rect of dirtyRects) {
		if (
			maxX >= rect.minX &&
			minX <= rect.maxX &&
			maxY >= rect.minY &&
			minY <= rect.maxY
		) {
			return true;
		}
	}
	return false;
}
export function resolveWebGPUTarget(
	context: WebGPUScreenPostProcessContext
): IRenderTexture {
	const output = context.resources.color.output;
	if (!output) {
		throw new Error("WebGPU post-process pass has no assigned color output.");
	}
	return output;
}
export interface ResolvedWebGLTarget {
	readonly source: WebGLTexture;
	readonly texture: WebGLTexture;
}

export function resolveWebGLTarget(
	context: WebGLScreenPostProcessContext | undefined
): ResolvedWebGLTarget | null {
	if (
		!context?.postFramebuffer ||
		!context.sceneColorTexture ||
		!context.fullscreenVao
	) {
		return null;
	}
	const source = context.resources.color.input ?? context.getSourceTexture();
	if (!source) {
		return null;
	}
	const texture = context.resources.color.output;
	if (!texture) {
		return null;
	}
	return { source, texture };
}

export function bindWebGLPostTarget(
	context: WebGLScreenPostProcessContext,
	program: WebGLProgram,
	targetTexture: WebGLTexture
): void {
	const gl = context.gl;
	gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
	context.bindColorTarget(targetTexture);
	gl.viewport(0, 0, context.width, context.height);
	gl.useProgram(program);
	gl.bindVertexArray(context.fullscreenVao);
	gl.disable(gl.CULL_FACE);
	gl.disable(gl.DEPTH_TEST);
	gl.disable(gl.BLEND);
}
