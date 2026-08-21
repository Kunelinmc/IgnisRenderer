import assert from "node:assert/strict";

import { Renderer } from "../../../src/rendering/Renderer.ts";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { ParticleSystem } from "../../../src/particles/ParticleSystem.ts";
import {
	PARTICLE_SIM_DELTA_TIME_SECONDS_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "../../../src/pipeline/types.ts";
import { DefaultParticleSimulator } from "../../../src/simulation/particles/DefaultParticleSimulator.ts";
import {
	installNoopPostProcessAdapter,
} from "../../helpers/postprocess.mjs";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class StubBackend extends TestRenderBackend {
	constructor() {
		super();
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
		installNoopPostProcessAdapter(
			this,
			"stub"
		);
		this.frameScheduling = "always";
		this.sharedStages = [];
		this.executedStages = [];
		this.particleBatchCount = 0;
		this.particleSimulator = new DefaultParticleSimulator({
			backendTag: "stub",
		});
	}

	resize() {}

	getAttachments({ width, height }) {
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
		const renderer = new Renderer(canvas, backend, camera);
		renderer.features.worldMatrix = Matrix4.identity();
		renderer.postProcess.getPass("gamma")?.disable();
		renderer.features.enableReflection = false;
		renderer.features.enableEnvironment = false;

		const system = new ParticleSystem({
			position: { x: 0, y: 0, z: -5 },
			emit: {
				rate: 0,
				bursts: [{ time: 0, count: 2 }],
			},
			templates: [
				{
					lifetimeRange: [2, 2],
					speedRange: [0, 0],
					sizeRange: [0.5, 1],
					shape: { kind: "billboard" },
				},
			],
		});
		renderer.scene.add(system);

		await renderer.renderFrame(16);
		await renderer.renderFrame(32);

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
