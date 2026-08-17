import type { Material } from "../../materials/Material";
import { ShaderMaterial, type ShaderTargetMode } from "../../materials/ShaderMaterial";
import type { FrameContext } from "../../pipeline/types";

import {
	collectWebGLLights,
	type WebGLLightState,
} from "./WebGLLightCollector";
import type { WebGLSceneProgramRepository } from "./WebGLSceneProgramRepository";
import {
	getWebGLSceneDepthVariantKey,
	getWebGLSceneVariantKey,
	createWebGLShaderMaterialFallbackVariant,
	resolveWebGLBuiltinDepthVariant,
	resolveWebGLBuiltinSceneVariant,
	type WebGLSceneDepthVariantDescriptor,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import { isMaterialTransparentPass } from "../../materials/transparency";
import { WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH } from "./constants";
import type {
	WebGLProgramWarmupContributor,
	WebGLProgramWarmupRequest,
	WebGLProgramWarmupTask,
} from "./WebGLWarmupCoordinator";

export interface WebGLSceneProgramPlan {
	readonly lightState: WebGLLightState;
	readonly sceneVariants: ReadonlyMap<string, WebGLSceneVariantDescriptor>;
	readonly depthVariants: ReadonlyMap<string, WebGLSceneDepthVariantDescriptor>;
	readonly shaderMaterialFallbackVariantKeys: ReadonlySet<string>;
}

/** @internal Produces every exact scene source variant a frame may select. */
export function planWebGLScenePrograms(
	context: FrameContext,
	materials: readonly Material[],
	modes: readonly ShaderTargetMode[]
): WebGLSceneProgramPlan {
	const lightState = collectPlannerLightState(context);
	const sceneVariants = new Map<string, WebGLSceneVariantDescriptor>();
	const shaderMaterialFallbackVariantKeys = new Set<string>();
	for (const material of materials) {
		if (material instanceof ShaderMaterial) {
			for (const mode of modes) {
				const fallback = createWebGLShaderMaterialFallbackVariant(mode);
				const key = getWebGLSceneVariantKey(fallback);
				sceneVariants.set(key, fallback);
				shaderMaterialFallbackVariantKeys.add(key);
			}
			continue;
		}
		for (const mode of modes) {
			addVariantAlternatives(
				sceneVariants,
				context,
				material,
				mode,
				0,
				lightState,
				false,
			);
			if (mode === "mrt") {
				addVariantAlternatives(
					sceneVariants,
					context,
					material,
					mode,
					0,
					lightState,
					true,
				);
			}
		}
		if (context.features?.enableOIT && isMaterialTransparentPass(material)) {
			addVariantAlternatives(
				sceneVariants,
				context,
				material,
				"single",
				1,
				lightState,
				false,
			);
			addVariantAlternatives(
				sceneVariants,
				context,
				material,
				"single",
				2,
				lightState,
				false,
			);
		}
	}

	const depthVariants = new Map<string, WebGLSceneDepthVariantDescriptor>();
	for (const material of materials) {
		if (
			material instanceof ShaderMaterial ||
			isMaterialTransparentPass(material) ||
			material.depthWrite === false
		) {
			continue;
		}
		const variant = resolveWebGLBuiltinDepthVariant(material);
		if (variant) {
			depthVariants.set(getWebGLSceneDepthVariantKey(variant), variant);
		}
	}

	return {
		lightState,
		sceneVariants,
		depthVariants,
		shaderMaterialFallbackVariantKeys,
	};
}

/** @internal Resolves one draw-time variant through the shared planner policy. */
export function resolveWebGLSceneDrawVariant(
	context: FrameContext,
	material: Material,
	mode: ShaderTargetMode,
	oitPassMode: 0 | 1 | 2,
	lightState: WebGLLightState | null,
	shadowTransmittanceAvailable: boolean,
	materialGBuffer: boolean
): WebGLSceneVariantDescriptor | null {
	return resolveWebGLBuiltinSceneVariant(
		context,
		material,
		mode,
		oitPassMode,
		{
			lightState,
			enableShadowTransmittance: shadowTransmittanceAvailable,
			enableIrradianceProbeGrid: true,
		},
		materialGBuffer,
	);
}

/** @internal Adapts scene planning and repository work into warmup tasks. */
export class WebGLSceneProgramWarmupContributor
	implements WebGLProgramWarmupContributor
{
	private readonly _repository: WebGLSceneProgramRepository;
	private readonly _enableEarlyZPrepass: boolean;

	public constructor(
		repository: WebGLSceneProgramRepository,
		enableEarlyZPrepass: boolean,
	) {
		this._repository = repository;
		this._enableEarlyZPrepass = enableEarlyZPrepass;
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		const modes: ShaderTargetMode[] =
			request.plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		const programPlan = planWebGLScenePrograms(
			request.context,
			request.plan.materials,
			modes,
		);
		const tasks: WebGLProgramWarmupTask[] = [{
			label: "WebGLSceneSource:builtin",
			priority: "core",
			run: () => this._repository.prepareBuiltinSceneVariants(
				programPlan.sceneVariants.values(),
			),
		}];

		for (const [key, variant] of programPlan.sceneVariants) {
			if (programPlan.shaderMaterialFallbackVariantKeys.has(key)) continue;
			tasks.push({
				label: `WebGLSceneProgram:builtin:${key}`,
				priority: "core",
				run: () => this._repository.warmupSceneProgram(
					undefined,
					variant.output,
					variant,
				),
			});
		}
		if (this._enableEarlyZPrepass) {
			for (const [key, variant] of programPlan.depthVariants) {
				tasks.push({
					label: `WebGLSceneDepthPrepassProgram:builtin:${key}`,
					priority: "core",
					run: () => this._repository.warmupSceneDepthPrepassProgram(
						undefined,
						"single",
						variant,
					),
				});
			}
		}

		for (const material of request.plan.materials) {
			if (!(material instanceof ShaderMaterial)) continue;
			for (const mode of modes) {
				tasks.push({
					label: `WebGLSceneProgram:material:${material.shaderId}:${mode}`,
					priority: "core",
					run: () => this._repository.warmupSceneProgram(material, mode),
				});
				if (this._enableEarlyZPrepass) {
					tasks.push({
						label:
							`WebGLSceneDepthPrepassProgram:material:` +
							`${material.shaderId}:${mode}`,
						priority: "core",
						run: () =>
							this._repository.warmupSceneDepthPrepassProgram(
								material,
								mode,
							),
					});
				}
			}
		}
		return tasks;
	}
}

function addVariantAlternatives(
	variants: Map<string, WebGLSceneVariantDescriptor>,
	context: FrameContext,
	material: Material,
	mode: ShaderTargetMode,
	oitPassMode: 0 | 1 | 2,
	lightState: WebGLLightState,
	materialGBuffer: boolean
): void {
	for (const transmittanceAvailable of [false, true]) {
		const variant = resolveWebGLSceneDrawVariant(
			context,
			material,
			mode,
			oitPassMode,
			lightState,
			transmittanceAvailable,
			mode === "mrt" && materialGBuffer,
		);
		if (variant) variants.set(getWebGLSceneVariantKey(variant), variant);
	}
}

function collectPlannerLightState(context: FrameContext): WebGLLightState {
	const environment = context.scene?.environment;
	return collectWebGLLights(
		context.scene?.lights ?? [],
		{
			enableLighting: context.features?.enableLighting ?? false,
			enableShadows: context.features?.enableShadows ?? false,
			shadowPlan: context.shadowPlan,
			enableSH: context.features?.enableSH ?? false,
			environmentTexture:
				environment?.lightingEnabled ? environment.iblTexture : null,
			enableClusteredLighting:
				context.features?.enableClusteredLighting ?? false,
			cameraWorldPosition: context.viewCamera?.getWorldPosition ?
				context.viewCamera.getWorldPosition(
					WEBGL_REFLECTION_PROBE_CAMERA_WORLD_POSITION_SCRATCH,
				)
			: null,
		},
	);
}
