import type { Material } from "../../materials/Material";
import { ShaderMaterial, type ShaderTargetMode } from "../../materials/ShaderMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";

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
	resolveWebGLPacketDeformationProfile,
	type WebGLDeformationProfile,
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
import type {
	WebGLMaterialSnapshotCache,
	WebGLResolvedMaterialSnapshot,
} from "./WebGLMaterialSnapshotCache";
import type { WebGLResolvedMaterialState } from "./WebGLMaterialState";

export interface WebGLSceneProgramPlan {
	readonly lightState: WebGLLightState;
	readonly sceneVariants: ReadonlyMap<string, WebGLSceneVariantDescriptor>;
	readonly depthVariants: ReadonlyMap<string, WebGLSceneDepthVariantDescriptor>;
}

/** @internal Produces every exact scene source variant a frame may select. */
export function planWebGLScenePrograms(
	context: FrameContext,
	inputs: readonly (Material | DrawPacket)[],
	modes: readonly ShaderTargetMode[],
	materialSnapshots?: WebGLMaterialSnapshotCache,
): WebGLSceneProgramPlan {
	const lightState = collectPlannerLightState(context);
	const sceneVariants = new Map<string, WebGLSceneVariantDescriptor>();
	const entries = inputs.map((input) => isDrawPacket(input) ? {
		material: input.submission.material.effective,
		deformation: resolveWebGLPacketDeformationProfile(input),
	} : {
		material: input,
		deformation: { skinProfile: "static", morphSemanticMask: 0 } as const,
	}).map((entry) => ({
		...entry,
		materialSnapshot:
			entry.material instanceof ShaderMaterial ? null
			: materialSnapshots?.resolve(entry.material) ?? null,
	}));
	for (const { material, deformation, materialSnapshot } of entries) {
		if (material instanceof ShaderMaterial) {
			for (const mode of modes) {
				const fallback = createWebGLShaderMaterialFallbackVariant(
					mode,
					deformation,
				);
				const key = getWebGLSceneVariantKey(fallback);
				sceneVariants.set(key, fallback);
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
				deformation,
				materialSnapshot,
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
					deformation,
					materialSnapshot,
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
				deformation,
				materialSnapshot,
			);
			addVariantAlternatives(
				sceneVariants,
				context,
				material,
				"single",
				2,
				lightState,
				false,
				deformation,
				materialSnapshot,
			);
		}
	}

	const depthVariants = new Map<string, WebGLSceneDepthVariantDescriptor>();
	for (const { material, deformation, materialSnapshot } of entries) {
		if (
			material instanceof ShaderMaterial ||
			isMaterialTransparentPass(material) ||
			material.depthWrite === false
		) {
			continue;
		}
		const variant = resolveWebGLBuiltinDepthVariant(
			material,
			deformation,
			materialSnapshot?.data,
		);
		if (variant) {
			depthVariants.set(getWebGLSceneDepthVariantKey(variant), variant);
		}
	}

	return {
		lightState,
		sceneVariants,
		depthVariants,
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
	materialGBuffer: boolean,
	deformation: WebGLDeformationProfile = {
		skinProfile: "static",
		morphSemanticMask: 0,
	},
	materialState?: WebGLResolvedMaterialState,
	materialVariant?: WebGLSceneVariantDescriptor["material"],
): WebGLSceneVariantDescriptor | null {
	if (material instanceof ShaderMaterial) {
		return createWebGLShaderMaterialFallbackVariant(mode, deformation);
	}
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
		deformation,
		materialState,
		materialVariant,
	);
}

/** @internal Adapts scene planning and repository work into warmup tasks. */
export class WebGLSceneProgramWarmupContributor
	implements WebGLProgramWarmupContributor
{
	private readonly _repository: WebGLSceneProgramRepository;
	private readonly _enableEarlyZPrepass: boolean;
	private readonly _materialSnapshots: WebGLMaterialSnapshotCache;

	public constructor(
		repository: WebGLSceneProgramRepository,
		enableEarlyZPrepass: boolean,
		materialSnapshots: WebGLMaterialSnapshotCache,
	) {
		this._repository = repository;
		this._enableEarlyZPrepass = enableEarlyZPrepass;
		this._materialSnapshots = materialSnapshots;
	}

	public collectWarmupTasks(
		request: WebGLProgramWarmupRequest,
	): readonly WebGLProgramWarmupTask[] {
		const modes: ShaderTargetMode[] =
			request.plan.sceneTargetMode === "mrt" ? ["mrt", "single"] : ["single"];
		const programPlan = planWebGLScenePrograms(
			request.context,
			[
				...(request.context.scene?.opaquePackets ?? []),
				...(request.context.scene?.transparentPackets ?? []),
			],
			modes,
			this._materialSnapshots,
		);
		const tasks: WebGLProgramWarmupTask[] = [{
			label: "WebGLSceneSource:builtin",
			priority: "core",
			run: () => this._repository.prepareBuiltinSceneVariants(
				programPlan.sceneVariants.values(),
			),
		}];

		for (const [key, variant] of programPlan.sceneVariants) {
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
	materialGBuffer: boolean,
	deformation: WebGLDeformationProfile,
	materialSnapshot: WebGLResolvedMaterialSnapshot | null,
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
			deformation,
			materialSnapshot?.data,
			materialSnapshot?.materialVariant,
		);
		if (variant) variants.set(getWebGLSceneVariantKey(variant), variant);
	}
}

function isDrawPacket(value: Material | DrawPacket): value is DrawPacket {
	return "submission" in value;
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
