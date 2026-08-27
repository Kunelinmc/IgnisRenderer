import assert from "node:assert/strict";

import { RenderPipelineRegistry } from "../../../src/pipeline/RenderPipelineRegistry.ts";
import {
	resolveFramePassRequirements,
} from "../../../src/pipeline/FramePassRequirements.ts";
import {
	DEFAULT_PIPELINE_STAGES,
} from "../../../src/pipeline/defaultPipeline.ts";
import { createTransientStore } from "../../../src/foundation/TransientStore.ts";
import {
	hasPostProcessExecutionPasses,
} from "../../../src/postprocess/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import { AlphaMode, Material } from "../../../src/materials/Material.ts";
import { EMPTY_SHADOW_FRAME_PLAN } from "../../../src/lights/shadows/ShadowFramePlan.ts";

function createShadowPlan(hasRasterWork) {
	return { ...EMPTY_SHADOW_FRAME_PLAN, hasRasterWork };
}

function createFrame(overrides = {}) {
	return {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		lights: [],
		particleSystems: [],
		hasActiveAnimations: false,
		camera: null,
		shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
		opaquePackets: [{}],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		decalPackets: [],
		...overrides,
	};
}

function createFramePlan(frame, features, postProcess, options = {}) {
	const registry =
		options.registry ??
		new RenderPipelineRegistry({
			stages: DEFAULT_PIPELINE_STAGES,
		});
	const stageOrder =
		options.stageOrder ??
		registry.getExecutionOrder(
			{
				hasActiveAnimations: frame.hasActiveAnimations,
				hasParticleSystems: (frame.particleSystems?.length ?? 0) > 0,
			},
			() => {},
		);
	const requirements =
		options.requirements ??
		resolveFramePassRequirements({
			frame,
			features,
			support: {
				meshParticles: options.backendCapabilities?.meshParticles === true,
				postProcess: options.backendCapabilities?.postProcess === true,
			},
			hasPostProcessWork:
				options.backendCapabilities?.postProcess === true &&
				hasPostProcessExecutionPasses(postProcess, {
					backend: options.backendType,
					frameContext: options.frameContext,
				}),
		});

	return registry.createFramePlan({
		stageOrder,
		frame,
		features,
		postProcess,
		transient: options.transient ?? createTransientStore(),
		requirements,
		backendType: options.backendType,
		backendCapabilities: options.backendCapabilities,
		incremental: options.incremental,
		frameContext: options.frameContext,
		incrementalStartStageIndex: options.incrementalStartStageIndex,
	});
}

function createBackendPasses(frame, features, postProcess) {
	return Array.from(createFramePlan(frame, features, postProcess).backendPasses);
}

function run() {
	const baseResolved = {
		enableLighting: true,
		enableSH: false,
		enableShadows: true,
		enableReflection: false,
		enableEnvironment: false,
		enableClusteredLighting: false,
		warnings: [],
		clusteredLightingOptions: {},
	};
	const createPostProcess = (overrides = {}) =>
		createResolvedPostProcess(overrides, "test");
	const backendCapabilities = {
		sh: false,
		shadows: true,
		reflection: true,
		environment: false,
		clusteredLighting: false,
		oit: false,
		postProcess: true,
		meshParticles: true,
	};

	const frame = createFrame({
		particleSystems: [{}],
		shadowCasterPackets: [{}],
		transparentPackets: [{}],
		reflectivePackets: [{}],
		shadowPlan: createShadowPlan(true),
	});
	const framePlan = createFramePlan(
		frame,
		{
			...baseResolved,
			enableReflection: true,
		},
		createPostProcess({
			ssao: { enabled: true },
			ssgi: { enabled: true },
			taa: { enabled: true },
			ssr: { enabled: true },
			volumetric: { enabled: true },
			fog: { enabled: true },
			bloom: { enabled: true },
			fxaa: { enabled: true },
		}),
		{
			backendType: "test",
			backendCapabilities,
		}
	);
	const plan = framePlan.backendPasses;

	assert.ok(framePlan.stageOrder.some((stage) => stage.id === "postprocess"));
	assert.equal(plan.some((pass) => pass.stage === "postprocess"), true);

	assert.deepEqual(
		plan.map((pass) => pass.stage),
		[
			"particle-sim",
			"shadow",
			"render-target-views",
			"reflection",
			"main-opaque",
			"main-transparent",
			"particles",
			"postprocess",
		]
	);
	assert.equal(
		plan.find((pass) => pass.stage === "particle-sim")?.enabled,
		true
	);
	assert.equal(plan.find((pass) => pass.stage === "shadow")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "reflection")?.enabled, true);
	assert.equal(
		plan.find((pass) => pass.stage === "main-opaque")?.enabled,
		true
	);
	assert.equal(
		plan.find((pass) => pass.stage === "main-transparent")?.enabled,
		true
	);
	assert.equal(plan.find((pass) => pass.stage === "particles")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "postprocess")?.enabled, true);
	assert.deepEqual(
		plan.find((pass) => pass.stage === "postprocess")?.dependsOn,
		["particles"]
	);
	assert.ok(
		framePlan.stageOrder.findIndex((stage) => stage.id === "particles") <
			framePlan.stageOrder.findIndex((stage) => stage.id === "postprocess")
	);
	assert.ok(
		framePlan.stageOrder.findIndex((stage) => stage.id === "postprocess") <
			framePlan.stageOrder.findIndex((stage) => stage.id === "sync-out")
	);

	const disabledPlan = createBackendPasses(
		createFrame(),
		baseResolved,
		createPostProcess({
			tonemap: { enabled: false },
			gamma: { enabled: false },
		})
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "particle-sim"),
		undefined
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "shadow")?.enabled,
		false
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "main-transparent")?.enabled,
		false
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "reflection")?.enabled,
		false
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "particles")?.enabled,
		false
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "postprocess")?.enabled,
		false
	);
	assert.equal(
		plan.find((pass) => pass.stage === "particle-sim")?.executor,
		"backend"
	);
	assert.deepEqual(
		plan.find((pass) => pass.stage === "main-opaque")?.dependsOn,
		["reflection", "shadow"]
	);

	const transparentMeshMaterial = new Material({
		name: "transparent-mesh-particle",
		alphaMode: AlphaMode.Blend,
		opacity: 0.5,
	});
	const meshParticleFrame = createFrame({
		particleSystems: [
			{
				visible: true,
				templates: [
					{
						castShadows: true,
						shape: {
							kind: "mesh",
							mesh: {
								primitives: [
									{
										material: transparentMeshMaterial,
										visible: true,
										castShadows: true,
									},
								],
							},
						},
					},
				],
			},
		],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		shadowPlan: createShadowPlan(true),
	});
	const meshParticlePlan = createFramePlan(
		meshParticleFrame,
		baseResolved,
		createPostProcess(),
		{
			backendType: "custom-mesh-backend",
			backendCapabilities,
		},
	).backendPasses;
	assert.equal(
		meshParticlePlan.find((pass) => pass.stage === "main-transparent")?.enabled,
		true,
	);
	assert.equal(
		meshParticlePlan.find((pass) => pass.stage === "shadow")?.enabled,
		true,
	);
	const unsupportedMeshParticlePlan = createFramePlan(
		{ ...meshParticleFrame, shadowPlan: createShadowPlan(false) },
		baseResolved,
		createPostProcess(),
		{
			backendType: "webgpu",
			backendCapabilities: {
				...backendCapabilities,
				meshParticles: false,
			},
		},
	).backendPasses;
	assert.equal(
		unsupportedMeshParticlePlan.find(
			(pass) => pass.stage === "main-transparent",
		)?.enabled,
		false,
	);
	assert.equal(
		unsupportedMeshParticlePlan.find((pass) => pass.stage === "shadow")
			?.enabled,
		false,
	);
	const explicitRequirementsPlan = createFramePlan(
		createFrame(),
		baseResolved,
		createPostProcess(),
		{
			backendType: "software",
			backendCapabilities: {
				...backendCapabilities,
				meshParticles: false,
				postProcess: false,
			},
			requirements: {
				requiredPasses: new Set([
					"main-opaque",
					"main-transparent",
					"postprocess",
				]),
			},
		},
	).backendPasses;
	assert.equal(
		explicitRequirementsPlan.find(
			(pass) => pass.stage === "main-transparent",
		)?.enabled,
		true,
	);
	assert.equal(
		explicitRequirementsPlan.find((pass) => pass.stage === "postprocess")
			?.enabled,
		true,
	);

	const particleCasterPlan = createBackendPasses(
		createFrame({
			particleSystems: [
				{
					visible: true,
					castShadows: true,
					blendMode: ParticleBlendMode.Alpha,
					shadowDensity: 1,
				},
			],
			shadowCasterPackets: [],
			shadowPlan: createShadowPlan(true),
		}),
		baseResolved,
		createPostProcess()
	);
	assert.equal(
		particleCasterPlan.find((pass) => pass.stage === "shadow")?.enabled,
		true
	);

	const additiveCasterPlan = createBackendPasses(
		createFrame({
			particleSystems: [
				{
					visible: true,
					castShadows: true,
					blendMode: ParticleBlendMode.Additive,
					shadowDensity: 1,
				},
			],
			shadowCasterPackets: [],
			shadowPlan: createShadowPlan(false),
		}),
		baseResolved,
		createPostProcess()
	);
	assert.equal(
		additiveCasterPlan.find((pass) => pass.stage === "shadow")?.enabled,
		false
	);

	const sceneFogPlan = createFramePlan(
		createFrame(),
		baseResolved,
		createPostProcess({
			fog: {
				enabled: true,
				options: {
					application: "scene",
				},
			},
		})
	);
	assert.ok(sceneFogPlan.stageOrder.some((stage) => stage.id === "postprocess"));
	assert.equal(
		sceneFogPlan.backendPasses.find((pass) => pass.stage === "postprocess")
			?.enabled,
		false
	);

	const registry = new RenderPipelineRegistry({
		stages: DEFAULT_PIPELINE_STAGES,
	});
	const defaultStageOrder = registry
		.getExecutionOrder(
			{ hasActiveAnimations: true, hasParticleSystems: false },
			() => {},
		)
		.map((stage) => stage.id);
	assert.ok(
		defaultStageOrder.indexOf("deformation-update") >
			defaultStageOrder.indexOf("csg-resolve"),
	);
	assert.ok(
		defaultStageOrder.indexOf("prepared-scene-build") >
			defaultStageOrder.indexOf("deformation-update"),
	);
	registry.registerPipelineStage({
		id: "custom-plan-pass",
		kind: "backend-pass",
		dependsOn: ["main-opaque"],
		shouldRun: () => true,
	});
	const customPlan = createFramePlan(
		createFrame(),
		baseResolved,
		createPostProcess(),
		{ registry }
	);
	const customBackendStages = customPlan.backendPasses.map((pass) => pass.stage);
	assert.ok(customBackendStages.includes("custom-plan-pass"));
	assert.ok(
		customBackendStages.indexOf("custom-plan-pass") >
			customBackendStages.indexOf("main-opaque")
	);

	console.log("Frame pass planning tests passed");
}

run();
