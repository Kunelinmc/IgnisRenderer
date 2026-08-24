import assert from "node:assert/strict";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { TestRenderBackend } from "../../helpers/TestRenderBackend.mjs";

class SHTestBackend extends TestRenderBackend {
	constructor() {
		super();
		this.type = "sh-test";
		this.capabilities.sh = true;
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

function createRenderer() {
	const backend = new SHTestBackend();
	const canvas = {
		width: 64,
		height: 64,
		getBoundingClientRect() {
			return { width: 64, height: 64 };
		},
	};
	const renderer = new Renderer(canvas, backend);
	renderer.features.enableShadows = false;
	renderer.features.enableReflection = false;
	renderer.features.enableEnvironment = false;
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

await testSHConfigurationInvalidatesAndComputesOnce();
console.log("Renderer SH configuration tests passed");
