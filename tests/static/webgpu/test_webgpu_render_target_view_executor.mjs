import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { WebGPURenderTargetViewExecutor } from
	"../../../src/backends/webgpu/WebGPURenderTargetViewExecutor.ts";
import { EMPTY_SHADOW_FRAME_PLAN } from
	"../../../src/lights/shadows/ShadowFramePlan.ts";
import {
	RenderTargetRegistrySnapshot,
} from "../../../src/rendering/CustomRenderTargets.ts";

const camera = new Camera();
const scene = {
	sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
	lights: [],
	particleSystems: [],
	hasActiveAnimations: false,
	environment: {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	},
	meshInstances: [],
	shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
	submissions: [],
	shadowCasterSubmissions: [],
	shadowTransmitterSubmissions: [],
	camera,
	opaquePackets: [],
	transparentPackets: [],
	shadowCasterPackets: [],
	shadowTransmitterPackets: [],
	reflectivePackets: [],
	decalPackets: [],
	occlusion: null,
	spatialIndex: null,
};

const context = {
	backendProfile: { id: "webgpu", capabilities: {}, frameScheduling: "always", lighting: {} },
	presentationAlphaMode: "opaque",
	viewCamera: camera,
	attachments: { width: 4, height: 4 },
	features: {
		enableEnvironment: false,
		enableReflection: true,
		enableShadows: false,
		warnings: [],
	},
	postProcess: { withPassDisabled() { return this; } },
	renderTargets: new RenderTargetRegistrySnapshot(),
	shadowPlan: EMPTY_SHADOW_FRAME_PLAN,
	scene,
	shCoeffs: SH.empty(),
	shAmbientCoeffs: SH.empty(),
	worldMatrix: Matrix4.identity(),
	incremental: { enabled: false, forceFullFrame: true, dirtyRects: [] },
	transient: new Map(),
};

async function testRecordsSingleColorViewWithoutSubmission() {
	const passes = [];
	let preparedContext = null;
	let destroyedScope = false;
	const resources = {
		createFrameScope() {
			return {
				prepare(nextContext) {
					preparedContext = nextContext;
					return {};
				},
				updateParticleShadowVolumes() {},
				destroy() { destroyedScope = true; },
			};
		},
		async buildClusteredLighting() {},
		async getEnvironmentResources() { return null; },
		async getDrawResources() { throw new Error("No packets should be drawn."); },
		getParticleBillboardRenderer() {
			return { async renderParticles() { throw new Error("Particles are disabled."); } };
		},
	};
	const host = {
		createTexture() { throw new Error("Target depth is supplied."); },
	};
	const executor = new WebGPURenderTargetViewExecutor(host, resources);
	const encoder = {
		beginRenderPass(descriptor) { passes.push(descriptor); },
		endRenderPass() {},
	};
	await executor.execute(
		encoder,
		context,
		{
			id: "job-1",
			targetId: "target-1",
			generation: 1,
			recurring: false,
			descriptor: {
				kind: "scene-view",
				camera,
				content: { environment: false, particles: false, shadows: "disabled" },
			},
			scene,
		},
		{
			id: "target-1",
			width: 4,
			height: 4,
			sampleCount: 1,
			color: [{ texture: { label: "color" }, format: "rgba16float", resolveTexture: null }],
			depth: { texture: { label: "depth" }, format: "depth32float", resolveTexture: null },
		},
	);
	assert.equal(passes.length, 1);
	assert.equal(passes[0].colorAttachments.length, 1);
	assert.strictEqual(preparedContext.viewCamera, camera);
	assert.equal(preparedContext.features.enableReflection, false);
	assert.equal(
		destroyedScope,
		false,
		"view resources must remain alive until queue submission",
	);
	executor.releaseSubmittedScopes();
	assert.equal(destroyedScope, true);
}

await testRecordsSingleColorViewWithoutSubmission();
console.log("WebGPU render-target view executor tests passed");
