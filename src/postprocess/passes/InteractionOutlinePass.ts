import { INTERACTION_TRANSIENT_STATE_KEY } from "../../pipeline/types";
import { clamp, sRGBToLinear } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import {
	computeInteractionOutlineShapeDistance,
	getInteractionOutlineShapeCode,
	resolveInteractionOutlineShape,
} from "../../interaction/outlineShape";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_INTERACTION_OUTLINE_LAYOUT as INTERACTION_OUTLINE_LAYOUT,
} from "../../renderers/webgpu/bufferLayouts";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import {
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	publishWebGPUColorTarget,
	resolveSoftwareDirtyRects,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	softwareRectIntersectsDirtyRects,
	type EmptyOptions,
	type IncrementalDirtyRect,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

export const INTERACTION_OUTLINE_PASS_ID = "interaction-outline";
export const INTERACTION_OUTLINE_PASS_ORDER = {
	id: INTERACTION_OUTLINE_PASS_ID,
	placement: "overlay",
	order: 800,
	incremental: {
		firstPass: "interaction-outline",
		grade: "light",
		inflationRadius: 2,
	},
} as const satisfies PostProcessPassMetadata;
export type WebGPUInteractionOutlineContext = WebGPURuntimePostProcessContext;
export type WebGLInteractionOutlineContext = WebGLScreenPostProcessContext;

/** @internal Software implementation for the built-in interaction outline pass. */
export class SoftwareInteractionOutlineImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "interaction-outline:software";

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const frameContext = request.frameContext;
		const state = frameContext.transient.get(INTERACTION_TRANSIENT_STATE_KEY);
		const pixels = frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		if (!state || state.selectedEntityIds.length === 0) {
			return { ran: true };
		}
		const width = Math.max(1, frameContext.attachments.width);
		const height = Math.max(1, frameContext.attachments.height);
		const dirtyRects = resolveSoftwareDirtyRects(frameContext);
		const outlineColor = state.outline?.color ?? { r: 255, g: 196, b: 64, a: 1 };
		const alpha = clamp(
			(state.outline?.opacity ?? 0.9) *
				(typeof outlineColor.a === "number" ? outlineColor.a : 1),
			0,
			1
		);
		const thickness = Math.max(1, Math.round(state.outline?.thickness ?? 2));
		const outlineShape = resolveInteractionOutlineShape(state.outline?.shape);
		const circles = collectProjectedOutlineCircles(
			frameContext,
			state.selectedEntityIds
		);
		for (const circle of circles) {
			const circleMinX = Math.max(
				0,
				Math.floor(circle.centerX - circle.radius - thickness)
			);
			const circleMinY = Math.max(
				0,
				Math.floor(circle.centerY - circle.radius - thickness)
			);
			const circleMaxX = Math.min(
				width - 1,
				Math.ceil(circle.centerX + circle.radius + thickness)
			);
			const circleMaxY = Math.min(
				height - 1,
				Math.ceil(circle.centerY + circle.radius + thickness)
			);
			if (
				!softwareRectIntersectsDirtyRects(
					circleMinX,
					circleMinY,
					circleMaxX,
					circleMaxY,
					dirtyRects
				)
			) {
				continue;
			}
			drawSoftwareInteractionOutlineShape(
				pixels,
				width,
				height,
				circle.centerX,
				circle.centerY,
				circle.radius,
				thickness,
				outlineShape,
				outlineColor.r,
				outlineColor.g,
				outlineColor.b,
				alpha,
				dirtyRects
			);
		}
		return { ran: true };
	}
}
interface WebGPUInteractionOutlineResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramWriter: ReturnType<typeof INTERACTION_OUTLINE_LAYOUT.createWriter>;
}
/** @internal WebGPU implementation for the built-in interaction outline pass. */
export class WebGPUInteractionOutlineImplementation
	implements PostProcessPassImplementation<WebGPUInteractionOutlineContext, EmptyOptions>
{
	public readonly id = "interaction-outline:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUInteractionOutlineResources>();
	private _resourceSet = new Set<WebGPUInteractionOutlineResources>();

	public async warmup(
		context: WebGPUInteractionOutlineContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUInteractionOutlineContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runInteractionOutlineKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("interaction-outline-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"interaction outline pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"interaction outline shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"interaction outline params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("interaction-outline-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUInteractionOutlineResources>();
	}

	private async _runInteractionOutlineKernel(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUInteractionOutlineContext
	): Promise<boolean> {
		const interactionState = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		const selectedEntityIds = interactionState?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return false;
		}
		const circles = collectProjectedOutlineCircles(
			request.frameContext,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return false;
		}
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline || !resources.params) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
		const outlineColor = interactionState?.outline?.color ?? {
			r: 255,
			g: 196,
			b: 64,
			a: 1,
		};
		const colorScale =
			Math.max(outlineColor.r, outlineColor.g, outlineColor.b) > 1 ? 255 : 1;
		const opacity = clamp(
			finiteOr(interactionState?.outline?.opacity, 0.9) *
				finiteOr(outlineColor.a, 1),
			0,
			1
		);
		const thickness = Math.max(
			1,
			finiteOr(interactionState?.outline?.thickness, 2)
		);
		const shapeCode = getInteractionOutlineShapeCode(
			interactionState?.outline?.shape
		);
		const params = resources.paramWriter;
		params.clear();
		params.writeVec("invSize", [
			1 / Math.max(target.width, 1),
			1 / Math.max(target.height, 1),
		]);
		params.writeF32("opacity", opacity);
		params.writeF32("thickness", thickness);
		params.writeVec("color", [
			sRGBToLinear(clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)),
			sRGBToLinear(clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)),
			sRGBToLinear(clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)),
			1,
		]);
		params.writeF32("circleCount", circles.length);
		params.writeF32("shape", shapeCode);
		for (let index = 0; index < circles.length; index++) {
			const circle = circles[index];
			params.writeVec(["circles", index], [
				circle.centerX,
				circle.centerY,
				circle.radius,
				0,
			]);
		}
		context.shared.compute.writeBuffer(
			resources.params,
			params.toFloat32Array() as Float32Array<ArrayBuffer>
		);
		const binding = context.shared.getCachedBindGroup(
			`interaction-outline-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 2, resource: resources.params },
				{ binding: 3, resource: target },
			],
			"WebGPUInteractionOutline_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUInteractionOutline" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUInteractionOutlineResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramWriter: INTERACTION_OUTLINE_LAYOUT.createWriter(),
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.interactionOutline.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUInteractionOutlineShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUInteractionOutlinePipeline",
				compute: {
					module: resources.module,
					entryPoint: "csMain",
				},
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUInteractionOutlineParams",
				size: INTERACTION_OUTLINE_LAYOUT.byteSize,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}
/** @internal WebGL implementation for the built-in interaction outline pass. */
export class WebGLInteractionOutlineImplementation
	implements PostProcessPassImplementation<WebGLInteractionOutlineContext, EmptyOptions>
{
	public readonly id = "interaction-outline:webgl";
	private readonly _circleData = new Float32Array(
		MAX_INTERACTION_OUTLINE_CIRCLES * 4
	);

	public warmup(context: WebGLInteractionOutlineContext | undefined): void {
		context?.programs.warmupInteractionOutlineProgram();
	}

	public execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLInteractionOutlineContext | undefined
	): PostProcessPassResult {
		const state = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		const selectedEntityIds = state?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return { ran: false };
		}
		const circles = collectProjectedOutlineCircles(
			request.frameContext,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return { ran: false };
		}
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}

		let writeOffset = 0;
		for (const circle of circles) {
			this._circleData[writeOffset] = circle.centerX;
			this._circleData[writeOffset + 1] = circle.centerY;
			this._circleData[writeOffset + 2] = circle.radius;
			this._circleData[writeOffset + 3] = 0;
			writeOffset += 4;
		}

		const outlineColor = state?.outline?.color ?? { r: 255, g: 196, b: 64, a: 1 };
		const colorScale =
			Math.max(outlineColor.r, outlineColor.g, outlineColor.b) > 1 ? 255 : 1;
		const linearR = sRGBToLinear(
			clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)
		);
		const linearG = sRGBToLinear(
			clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)
		);
		const linearB = sRGBToLinear(
			clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)
		);
		const alpha = clamp(
			finiteOr(state?.outline?.opacity, 0.9) * finiteOr(outlineColor.a, 1),
			0,
			1
		);
		const thickness = Math.max(1, finiteOr(state?.outline?.thickness, 2));
		const shapeCode = getInteractionOutlineShapeCode(state?.outline?.shape);

		const gl = context.gl;
		const program = context.programs.tryGetInteractionOutlineProgram();
		if (!program) {
			return { ran: false };
		}
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.outlineColor) {
			gl.uniform4f(program.uniforms.outlineColor, linearR, linearG, linearB, 1);
		}
		if (program.uniforms.outlineParams) {
			gl.uniform3f(program.uniforms.outlineParams, alpha, thickness, shapeCode);
		}
		if (program.uniforms.viewportSize) {
			gl.uniform2f(
				program.uniforms.viewportSize,
				Math.max(1, context.width),
				Math.max(1, context.height)
			);
		}
		if (program.uniforms.circleCount) {
			gl.uniform1i(program.uniforms.circleCount, circles.length);
		}
		if (program.uniforms.circles) {
			gl.uniform4fv(
				program.uniforms.circles,
				this._circleData.subarray(0, circles.length * 4)
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}
/**
 * Stateful logical interaction outline pass.
 */
export class InteractionOutlinePass extends PostProcessPass<
	EmptyOptions,
	EmptyOptions
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			...INTERACTION_OUTLINE_PASS_ORDER,
			incremental:
				config.incremental ?? INTERACTION_OUTLINE_PASS_ORDER.incremental,
			warningLabel: "interaction outline",
			implementations: {
				software: new SoftwareInteractionOutlineImplementation(),
				webgpu: new WebGPUInteractionOutlineImplementation(),
				webgl: new WebGLInteractionOutlineImplementation(),
			},
		});
	}

	public override shouldExecute(
		request: PostProcessPassResolveRequest<EmptyOptions>
	): boolean {
		if (!request.frameContext) {
			return true;
		}
		const interactionState = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		return (interactionState?.selectedEntityIds?.length ?? 0) > 0;
	}
}
function drawSoftwareInteractionOutlineShape(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	centerX: number,
	centerY: number,
	radius: number,
	thickness: number,
	shape: "circle" | "square" | "diamond" | "octagon",
	red: number,
	green: number,
	blue: number,
	alpha: number,
	dirtyRects: IncrementalDirtyRect[]
): void {
	const minX = Math.max(0, Math.floor(centerX - radius - thickness));
	const minY = Math.max(0, Math.floor(centerY - radius - thickness));
	const maxX = Math.min(width - 1, Math.ceil(centerX + radius + thickness));
	const maxY = Math.min(height - 1, Math.ceil(centerY + radius + thickness));
	const inner = Math.max(0, radius - thickness * 0.5);
	const outer = radius + thickness * 0.5;

	for (let y = minY; y <= maxY; y++) {
		for (let x = minX; x <= maxX; x++) {
			if (!softwareRectIntersectsDirtyRects(x, y, x, y, dirtyRects)) {
				continue;
			}
			const dx = x + 0.5 - centerX;
			const dy = y + 0.5 - centerY;
			const shapeDistance = computeInteractionOutlineShapeDistance(
				dx,
				dy,
				shape
			);
			if (shapeDistance < inner || shapeDistance > outer) {
				continue;
			}
			const index = (y * width + x) << 2;
			const invAlpha = 1 - alpha;
			pixels[index] = Math.round(pixels[index] * invAlpha + red * alpha);
			pixels[index + 1] = Math.round(
				pixels[index + 1] * invAlpha + green * alpha
			);
			pixels[index + 2] = Math.round(
				pixels[index + 2] * invAlpha + blue * alpha
			);
			pixels[index + 3] = 255;
		}
	}
}
