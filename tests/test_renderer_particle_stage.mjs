import assert from "node:assert/strict";

import { Renderer } from "../src/renderers/Renderer.ts";
import { Camera } from "../src/cameras/Camera.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { ParticleSystem } from "../src/particles/ParticleSystem.ts";
import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "../src/pipeline/types.ts";
import { DefaultParticleSimulator } from "../src/simulation/particles/DefaultParticleSimulator.ts";
import { ALL_POST_PROCESS_CAPABILITIES } from "./helpers/postprocess.mjs";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
		};
		this.postProcess = {
			capabilities: ALL_POST_PROCESS_CAPABILITIES,
		};
		this.frameScheduling = "always";
		this.passExecutors = {
			shadow: "shared",
		};
		this.sharedStages = [];
		this.executedStages = [];
		this.particleBatchCount = 0;
		this.particleSimulator = new DefaultParticleSimulator({
			backendTag: "stub",
		});
	}

	async init() {}

	resize() {}

	getAttachments(width, height) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		};
	}

	beginFrame(context) {
		this.particleSimulator.beginFrame(context);
	}

	executeSharedPass(pass) {
		this.sharedStages.push(pass.stage);
	}

	executePass(pass, context) {
		this.executedStages.push(pass.stage);
		if (pass.stage === "particle-sim") {
			this.particleSimulator.simulate(
				context,
				context.transient.get(PARTICLE_SIM_DELTA_TIME_SECONDS_KEY) ?? 0
			);
			this.particleSimulator.emitRenderBatches(context);
		}
		if (pass.stage === "particles") {
			const batches =
				context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
			this.particleBatchCount = batches.length;
		}
	}

	endFrame() {
		this.particleSimulator.endFrame();
	}
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new StubBackend();
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const camera = new Camera();
		const renderer = new Renderer(backend, canvas, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.disable("gamma");
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;

		const system = new ParticleSystem({
			position: { x: 0, y: 0, z: -5 },
			emit: {
				rate: 0,
				bursts: [{ time: 0, count: 2 }],
				lifetimeRange: [2, 2],
				speedRange: [0, 0],
			},
		});
		renderer.scene.add(system);

		await renderer.renderScene(16);
		await renderer.renderScene(32);

		const incrementalStats = renderer.getLastIncrementalFrameStats();
		assert.equal(incrementalStats?.forceFullFrame, true);
		assert.equal(incrementalStats?.temporalHistoryReset, true);
		assert.equal(backend.sharedStages.includes("particle-sim"), false);
		assert.ok(backend.executedStages.includes("particle-sim"));
		assert.ok(backend.executedStages.includes("main-opaque"));
		assert.ok(backend.executedStages.includes("particles"));
		assert.ok(backend.particleBatchCount > 0);

		console.log("Renderer particle stage tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
