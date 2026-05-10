import assert from "node:assert/strict";
import { FramePlanner } from "../src/pipeline/FramePlanner.ts";
import { resolvePostProcessState } from "../src/pipeline/PostProcess.ts";
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
	const capabilities = {
		ssao: true,
		ssgi: true,
		taa: true,
		ssr: true,
		volumetric: true,
		fog: true,
		"motion-blur": true,
		dof: true,
		bloom: true,
		tonemap: true,
		"color-filter": true,
		fxaa: true,
		"interaction-outline": true,
		gamma: true,
	};
	const createPostProcess = (overrides = {}) =>
		resolvePostProcessState(overrides, capabilities, "test");

	const frame = createFrame({
		particleSystems: [{}],
		shadowCasterPackets: [{}],
		transparentPackets: [{}],
		reflectivePackets: [{}],
	});
	const plan = FramePlanner.build(
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

	assert.deepEqual(
		plan.map((pass) => pass.stage),
		[
			"animation-sim",
			"particle-sim",
			"shadow",
			"reflection",
			"main-opaque",
			"main-transparent",
			"particles",
			"ssao",
			"ssgi",
			"taa",
			"ssr",
			"volumetric",
			"fog",
			"motion-blur",
			"dof",
			"bloom",
			"tonemap",
			"color-filter",
			"fxaa",
			"gamma",
		]
	);
	assert.equal(
		plan.find((pass) => pass.stage === "particle-sim")?.enabled,
		true
	);
	assert.equal(
		plan.find((pass) => pass.stage === "animation-sim")?.enabled,
		false
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
	assert.equal(plan.find((pass) => pass.stage === "ssao")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "ssgi")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "taa")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "ssr")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "volumetric")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "fog")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "bloom")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "tonemap")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "fxaa")?.enabled, true);
	assert.equal(plan.find((pass) => pass.stage === "gamma")?.enabled, true);

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
		disabledPlan.find((pass) => pass.stage === "animation-sim")?.enabled,
		false
	);
	assert.equal(
		disabledPlan.find((pass) => pass.stage === "particle-sim")?.enabled,
		false
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

	const sceneFogPlan = FramePlanner.build(
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
	assert.equal(sceneFogPlan.find((pass) => pass.stage === "fog")?.enabled, false);

	console.log("Frame planner tests passed");
}

run();
