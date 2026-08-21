import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Node } from "../../../src/core/Node.ts";
import { PhysicsSystem } from "../../../src/physics/PhysicsSystem.ts";
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
		this.postAnimationSeenBeforeBegin = false;
		this._postAnimationFlag = false;
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

	beginFrame() {
		this.postAnimationSeenBeforeBegin = this._postAnimationFlag;
	}

	executePass() {}

	endFrame() {}
}

async function testPostAnimationHookOrder() {
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

		renderer.on("postanimation", () => {
			backend._postAnimationFlag = true;
		});

		await renderer.renderFrame(16);
		assert.equal(backend.postAnimationSeenBeforeBegin, true);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

function testPhysicsWakeupBridge() {
	const physics = new PhysicsSystem();
	physics.initSync();
	physics.createWorld({ worldId: "main", mode: "variable" });

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	let wakeups = 0;
	const disconnect = physics.connectRendererWakeup({
		requestRender() {
			wakeups++;
		},
	});
	physics.step(0.016);
	assert.ok(wakeups > 0);
	disconnect();
}

await testPostAnimationHookOrder();
testPhysicsWakeupBridge();
console.log("Renderer postanimation hook tests passed");
