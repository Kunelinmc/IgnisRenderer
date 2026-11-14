import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { SH } from "../../../src/maths/SH.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class SHTestBackend extends TestRenderBackend {
	constructor(localizedProbeMode = "accumulate-globally") {
		super();
		this.type = "sh-test";
		this.capabilities.sh = true;
		this.localizedProbeMode = localizedProbeMode;
	}

	get profile() {
		return { ...super.profile, lighting: { localizedProbeMode: this.localizedProbeMode } };
	}

	executePass(_pass, context) {
		this.frameContext = context;
	}
}

async function withDOMGlobals(callback) {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;
	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;
		await callback();
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

function createRenderer(localizedProbeMode) {
	const backend = new SHTestBackend(localizedProbeMode);
	const canvas = {
		width: 64,
		height: 64,
		getBoundingClientRect() {
			return { width: 64, height: 64 };
		},
	};
	const renderer = new Renderer(canvas, backend);
	renderer.features.enableReflection = false;
	renderer.scene.environment.backgroundEnabled = false;
	return renderer;
}

async function testSHConfigurationInvalidatesAndComputesOnce() {
	await withDOMGlobals(async () => {
		const renderer = createRenderer();
		const coordinator = renderer._coordinator;
		const updateSH = coordinator.updateSH.bind(coordinator);
		let updateSHCalls = 0;
		coordinator.updateSH = (...args) => {
			updateSHCalls++;
			return updateSH(...args);
		};

		try {
			await renderer.initialize();
			const initial = await renderer.renderFrame(1);
			assert.equal(initial.rendered, true);
			assert.equal(updateSHCalls, 0);

			updateSHCalls = 0;
			renderer.features.enableSH = true;
			const enabled = await renderer.renderFrame(2);
			assert.equal(enabled.rendered, true);
			assert.equal(updateSHCalls, 1);

			const probe = renderer.scene.add(new LightProbe({
				source: "capturedScene",
				includeEnvironment: false,
				includeMeshes: false,
			}));
			await renderer.renderFrame(3);
			probe.requestCapture();
			const requested = await renderer.renderFrame(4);
			assert.equal(requested.rendered, true);
		} finally {
			await renderer.destroy();
		}
	});
}

async function testFrameSHContainsOnlyProbeRadiance() {
	await withDOMGlobals(async () => {
		for (const mode of ["accumulate-globally", "backend-local"]) {
			const renderer = createRenderer(mode);
			const backend = renderer._backend;
			const globalSH = SH.empty();
			globalSH[0] = { r: 30, g: 20, b: 10 };
			globalSH[7] = { r: -3, g: 4, b: 6 };
			const localSH = SH.empty();
			localSH[0] = { r: 12, g: 6, b: 3 };
			renderer.features.enableSH = true;
			renderer.scene.add(new AmbientLight({ intensity: 10 }));
			renderer.scene.add(new DirectionalLight({
				color: { r: 128, g: 128, b: 128 }, intensity: 10,
			}));
			const probe = renderer.scene.add(new LightProbe({ sh: globalSH }));
			renderer.scene.add(new LightProbe({ sh: localSH, shape: "sphere" }));
			try {
				await renderer.initialize();
				await renderer.renderFrame(1);
				const firstFrame = backend.frameContext;
				const expectedDC = mode === "backend-local" ?
					{ r: 30, g: 20, b: 10 } : { r: 42, g: 26, b: 13 };
				assert.deepEqual(firstFrame.shAmbientCoeffs[0], expectedDC);
				assert.deepEqual(firstFrame.shAmbientCoeffs[7], { r: -3, g: 4, b: 6 });

				// Diagnostic reads must not let callers mutate the active frame's SH.
				const snapshot = renderer.shAmbientCoeffs;
				snapshot[0].r = 999;
				assert.deepEqual(renderer.shAmbientCoeffs[0], expectedDC);
				assert.deepEqual(firstFrame.shAmbientCoeffs[0], expectedDC);

				probe.sh[0].r = 60;
				await renderer.renderFrame(2);
				assert.equal(backend.frameContext.shAmbientCoeffs[0].r,
					mode === "backend-local" ? 60 : 72);
				assert.deepEqual(firstFrame.shAmbientCoeffs[0], expectedDC);

				renderer.features.enableSH = false;
				await renderer.renderFrame(3);
				assert.deepEqual(backend.frameContext.shAmbientCoeffs, SH.empty());
				assert.deepEqual(renderer.shAmbientCoeffs, SH.empty());
			} finally {
				await renderer.destroy();
			}
		}
	});
}

await testSHConfigurationInvalidatesAndComputesOnce();
await testFrameSHContainsOnlyProbeRadiance();
console.log("Renderer SH configuration tests passed");
