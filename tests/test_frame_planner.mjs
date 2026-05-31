import assert from "node:assert/strict";
import { FramePlanner } from "../src/pipeline/FramePlanner.ts";
import { RenderPipelineRegistry } from "../src/pipeline/RenderPipelineRegistry.ts";
import {
	createDefaultBackendPasses,
	createDefaultRendererStages,
} from "../src/pipeline/defaultPipeline.ts";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";
import { ParticleBlendMode } from "../src/particles/types.ts";

function createFrame(overrides = {}) {
	return {
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		lights: [],
		particleSystems: [],
		hasActiveAnimations: false,
		camera: null,
		shadowMaps: new Map(),
		opaquePackets: [{}],
		transparentPackets: [],
		shadowCasterPackets: [],
		shadowTransmitterPackets: [],
		reflectivePackets: [],
		...overrides,
	};
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

	const frame = createFrame({
		particleSystems: [{}],
		shadowCasterPackets: [{}],
		transparentPackets: [{}],
		reflectivePackets: [{}],
	});
	const framePlan = FramePlanner.buildFramePlan(
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
		})
	);
	const plan = framePlan.backendPasses;

	assert.ok(framePlan.stageOrder.some((stage) => stage.id === "postprocess"));
	assert.equal(plan.some((pass) => pass.stage === "postprocess"), false);

	assert.deepEqual(
		plan.map((pass) => pass.stage),
		[
			"particle-sim",
			"shadow",
			"reflection",
			"main-opaque",
			"main-transparent",
			"particles",
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
	assert.equal(plan.find((pass) => pass.stage === "postprocess"), undefined);

	const disabledPlan = FramePlanner.build(
		createFrame(),
		baseResolved,
		createPostProcess({
			tonemap: { enabled: false },
			gamma: { enabled: false },
			"interaction-outline": { enabled: false },
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
		disabledPlan.find((pass) => pass.stage === "postprocess"),
		undefined
	);
	assert.equal(
		plan.find((pass) => pass.stage === "particle-sim")?.executor,
		"backend"
	);

	const particleCasterPlan = FramePlanner.build(
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
		}),
		baseResolved,
		createPostProcess()
	);
	assert.equal(
		particleCasterPlan.find((pass) => pass.stage === "shadow")?.enabled,
		true
	);

	const additiveCasterPlan = FramePlanner.build(
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
		}),
		baseResolved,
		createPostProcess()
	);
	assert.equal(
		additiveCasterPlan.find((pass) => pass.stage === "shadow")?.enabled,
		false
	);

	const sceneFogPlan = FramePlanner.buildFramePlan(
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
		sceneFogPlan.backendPasses.some((pass) => pass.stage === "postprocess"),
		false
	);

	const registry = new RenderPipelineRegistry({
		stages: createDefaultRendererStages(),
		backendPasses: createDefaultBackendPasses(),
	});
	registry.registerBackendPass({
		id: "custom-plan-pass",
		dependsOn: ["main-opaque"],
		shouldRun: () => true,
	});
	const customPlan = FramePlanner.buildFramePlan(
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

	console.log("Frame planner tests passed");
}

run();
