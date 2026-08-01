import type { Material } from "../../materials/Material";
import { ShaderMaterial, type ShaderTargetMode } from "../../materials/ShaderMaterial";
import { isMaterialTransparentPass } from "../../materials/transparency";
import type { FrameContext } from "../../pipeline/types";
import {
	toShaderCompileError,
	type WarmupPhaseCounters,
	type WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { createWarmupYieldController } from "../../pipeline/WarmupScheduler";
import { ShaderSource, type WebGLSceneLightLimits } from "../../shaders/ShaderSource";
import type { WarmupOptions } from "../IRenderBackend";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";

import { WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH } from "./constants";
import { collectWebGLLights, type WebGLLightState } from "./WebGLLightCollector";
import type { WebGLPostProcessServices } from "./WebGLPostProcessServices";
import type { WebGLProgramCompiler, WebGLProgramWarmupHandle } from "./WebGLProgramCompiler";
import type { WebGLProgramLibrary } from "./WebGLProgramLibrary";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	resolveWebGLBuiltinDepthVariant,
	resolveWebGLBuiltinSceneVariant,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import {
	WebGLProgramWarmupQueue,
	type WebGLProgramWarmupPriority,
} from "./WebGLProgramWarmupQueue";
import type { BackendPostProcessRuntime } from "../../postprocess/BackendPostProcessRuntime";
import type { PostProcessPlan } from "../../postprocess/PostProcessPlanner";

export interface WebGLWarmupCoordinatorServices {
	getPrograms(): WebGLProgramLibrary;
	getCompiler(): WebGLProgramCompiler;
	readonly postProcessRuntime: BackendPostProcessRuntime;
	readonly postProcess: WebGLPostProcessServices;
	readonly enableEarlyZPrepass: boolean;
	readonly maxTextureImageUnits: number;
	readonly irradianceProbeGridSamplingSupported: boolean;
}

/** Compiles WebGL program variants without coupling warmup to frame execution. */
export class WebGLWarmupCoordinator {
	private readonly _services: WebGLWarmupCoordinatorServices;

	public constructor(services: WebGLWarmupCoordinatorServices) {
		this._services = services;
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan,
		options: WarmupOptions = {},
		postProcessPlan?: PostProcessPlan,
		signal?: AbortSignal | null,
	): Promise<WarmupPhaseCounters> {
		const yieldController = createWarmupYieldController(options);
		const queue = new WebGLProgramWarmupQueue();
		const enqueue = (
			label: string,
			priority: WebGLProgramWarmupPriority,
			action: () => unknown | Promise<unknown>,
		): void => {
			queue.enqueue({
				label,
				priority,
				action: () => this._collectWarmupHandles(action),
			});
		};

		const modes: ShaderTargetMode[] =
			plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		const lightState = this._collectLightState(context);
		const sceneVariants = this._collectSceneVariants(
			context,
			plan.materials,
			modes,
			lightState,
		);
		const depthVariants = this._collectDepthVariants(plan.materials);
		const limits = this._getSceneLightLimits();
		enqueue("WebGLSceneSource:builtin", "core", () =>
			ShaderSource.prepareMany(
				[...sceneVariants.values()].flatMap((variant) => [
					{ key: "webgl.scene.raw" as const, params: { limits, variant } },
					{ key: "webgl.scene.composite" as const, params: { limits, variant } },
				]),
			),
		);
		for (const [key, variant] of sceneVariants) {
			enqueue(`WebGLSceneProgram:builtin:${key}`, "core", () =>
				this._services.getPrograms().warmupSceneProgram(
					undefined,
					variant.output,
					variant,
				),
			);
		}
		if (this._services.enableEarlyZPrepass) {
			for (const [key, variant] of depthVariants) {
				enqueue(`WebGLSceneDepthPrepassProgram:builtin:${key}`, "core", () =>
					this._services.getPrograms().warmupSceneDepthPrepassProgram(
						undefined,
						"single",
						variant,
					),
				);
			}
		}
		for (const material of plan.materials) {
			if (!(material instanceof ShaderMaterial)) continue;
			for (const mode of modes) {
				enqueue(
					`WebGLSceneProgram:material:${material.shaderId}:${mode}`,
					"core",
					() => this._services.getPrograms().warmupSceneProgram(material, mode),
				);
				if (this._services.enableEarlyZPrepass) {
					enqueue(
						`WebGLSceneDepthPrepassProgram:material:${material.shaderId}:${mode}`,
						"core",
						() => this._services.getPrograms().warmupSceneDepthPrepassProgram(
							material,
							mode,
						),
					);
				}
			}
		}

		if (plan.enableEnvironment) {
			enqueue("WebGLEnvironmentProgram", "optional", () =>
				this._services.getPrograms().warmupEnvironmentProgram(),
			);
		}
		if (plan.enableShadows) {
			enqueue("WebGLShadowDepthProgram", "optional", () =>
				this._services.getPrograms().warmupShadowDepthProgram(),
			);
		}
		if (plan.enableParticles) {
			enqueue("WebGLParticleProgram", "optional", () =>
				this._services.getPrograms().warmupParticleProgram(),
			);
		}
		if (context.features?.enableOIT) {
			enqueue("WebGLOITResolveProgram", "optional", () =>
				this._services.getPrograms().warmupOITResolveProgram(),
			);
		}

		const graph = postProcessPlan ?? (plan.includePostProcess ?
			this._services.postProcessRuntime.planWarmup(context) : null);
		const warmed = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmed.has(passId)) continue;
			const compiled = graph?.passes.find((pass) => pass.id === passId);
			const implementation = compiled?.implementation;
			if (typeof implementation?.warmup !== "function") continue;
			warmed.add(passId);
			enqueue(`WebGLPostWarmup:${passId}`, "postprocess", async () => {
				const warmupContext =
					this._services.postProcess.getPassWarmupExecutionContext(
						compiled.id,
						compiled.declaration,
					);
				await implementation.warmup?.(warmupContext, {
					frameContext: context,
					postProcess: context.postProcess,
					backend: "webgl",
					context: warmupContext,
					options: compiled.options,
				});
			});
		}
		enqueue("WebGLPresentProgram", "core", () =>
			this._services.getPrograms().warmupPresentProgram(),
		);

		const result = await queue.run(yieldController, options, signal);
		return {
			phase: "webgl-programs",
			total: result.handles + result.enqueueFailures,
			compiled: result.compiled,
			skipped: 0,
			failed: result.enqueueFailures + result.failed,
			errors: result.errors.map((entry) =>
				toShaderCompileError(entry.error, "webgl", entry.label),
			),
		};
	}

	private _collectLightState(context: FrameContext): WebGLLightState {
		const environment = context.scene?.environment;
		return collectWebGLLights(
			context.scene?.lights ?? [],
			context.features?.enableLighting ?? false,
			context.features?.enableShadows ?? false,
			context.shadowMaps ?? new Map(),
			context.features?.enableSH ?? false,
			environment?.lightingEnabled ? environment.iblTexture : null,
			context.features?.enableClusteredLighting ?? false,
			context.viewCamera?.getWorldPosition ?
				context.viewCamera.getWorldPosition(
					WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
				)
			: null,
		);
	}

	private _collectSceneVariants(
		context: FrameContext,
		materials: readonly Material[],
		modes: readonly ShaderTargetMode[],
		lightState: WebGLLightState,
	): Map<string, WebGLSceneVariantDescriptor> {
		const variants = new Map<string, WebGLSceneVariantDescriptor>();
		for (const material of materials) {
			if (material instanceof ShaderMaterial) continue;
			for (const mode of modes) {
				this._addSceneVariant(variants, context, material, mode, 0, lightState);
			}
			if (context.features?.enableOIT && isMaterialTransparentPass(material)) {
				this._addSceneVariant(variants, context, material, "single", 1, lightState);
				this._addSceneVariant(variants, context, material, "single", 2, lightState);
			}
		}
		return variants;
	}

	private _addSceneVariant(
		variants: Map<string, WebGLSceneVariantDescriptor>,
		context: FrameContext,
		material: Material,
		mode: ShaderTargetMode,
		oitPassMode: 0 | 1 | 2,
		lightState: WebGLLightState,
	): void {
		const variant = resolveWebGLBuiltinSceneVariant(
			context,
			material,
			mode,
			oitPassMode,
			{
				lightState,
				enableShadowTransmittance:
					this._services.maxTextureImageUnits >= 17,
				enableIrradianceProbeGrid:
					this._services.irradianceProbeGridSamplingSupported,
			},
			mode === "mrt",
		);
		if (variant) variants.set(getWebGLSceneVariantKey(variant), variant);
	}

	private _collectDepthVariants(
		materials: readonly Material[],
	): Map<string, WebGLSceneDepthVariantDescriptor> {
		const variants = new Map<string, WebGLSceneDepthVariantDescriptor>();
		for (const material of materials) {
			if (
				material instanceof ShaderMaterial ||
				isMaterialTransparentPass(material) ||
				material.depthWrite === false
			) {
				continue;
			}
			const variant = resolveWebGLBuiltinDepthVariant(material);
			if (variant) variants.set(getWebGLSceneDepthVariantKey(variant), variant);
		}
		return variants;
	}

	private _getSceneLightLimits(): WebGLSceneLightLimits {
		return {
			maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
			maxPointLights: MAX_POINT_LIGHTS,
			maxSpotLights: MAX_SPOT_LIGHTS,
			enableShadowTransmittance: this._services.maxTextureImageUnits >= 17,
			enableIrradianceProbeGrid:
				this._services.irradianceProbeGridSamplingSupported,
		};
	}

	private async _collectWarmupHandles(
		action: () => unknown | Promise<unknown>,
	): Promise<WebGLProgramWarmupHandle[]> {
		const mark = this._services.getCompiler().markWarmupHandles();
		const result = await action();
		const logged = this._services.getCompiler().collectWarmupHandlesSince(mark);
		if (logged.length > 0) return logged;
		if (isWebGLProgramWarmupHandle(result)) return [result];
		if (Array.isArray(result) && result.every(isWebGLProgramWarmupHandle)) {
			return result;
		}
		return [];
	}
}

function isWebGLProgramWarmupHandle(
	value: unknown,
): value is WebGLProgramWarmupHandle {
	return (
		typeof value === "object" &&
		value !== null &&
		"label" in value &&
		typeof (value as { isComplete?: unknown }).isComplete === "function" &&
		typeof (value as { finalize?: unknown }).finalize === "function"
	);
}
