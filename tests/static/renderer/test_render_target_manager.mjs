import assert from "node:assert/strict";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import { RenderTargetManager } from "../../../src/rendering/CustomRenderTargets.ts";

function createManager() {
	return new RenderTargetManager({
		supportsJobs: true,
		invalidate() {},
		async readColor() {
			return {
				bytes: new Uint8Array(8),
				width: 1,
				height: 1,
				format: TextureFormat.RGBA16Float,
				bytesPerPixel: 8,
				bytesPerRow: 8,
				origin: "top-left",
				toFloat32: () => new Float32Array(4),
				toRGBAFloat32: () => new Float32Array(4),
				toNormalizedRGBA8Float32: () => new Float32Array(4),
			};
		},
	});
}

function createTarget(manager) {
	return manager.create({
		size: { mode: "fixed", width: 1, height: 1 },
		color: [{ format: TextureFormat.RGBA16Float }],
		depth: { format: TextureFormat.Depth32Float },
	});
}

async function testAbortRequeuesAndCommitPublishesGeneration() {
	const manager = createManager();
	const target = createTarget(manager);
	await assert.rejects(() => target.readColor(), /no committed generation/);
	const ticket = target.enqueueJob({ kind: "custom-pass", execute() {} });
	const first = manager.createJobSnapshot(() => null);
	assert.equal(first.size, 1);
	manager.abortFrame(first);
	const retry = manager.createJobSnapshot(() => null);
	assert.equal(retry.size, 1);
	await manager.commitFrame(retry);
	const completion = await ticket.done;
	assert.equal(completion.generation, 2);
	assert.equal(completion.readback, null);
	await target.readColor();
}

async function testDestroyRejectsInflightTicket() {
	const manager = createManager();
	const target = createTarget(manager);
	const ticket = target.enqueueJob({ kind: "custom-pass", execute() {} });
	manager.createJobSnapshot(() => null);
	target.destroy();
	await assert.rejects(ticket.done, /destroyed/);
}

function testRecurringJobsRemainRegistered() {
	const manager = createManager();
	const target = createTarget(manager);
	const registration = target.registerJob({ kind: "custom-pass", execute() {} });
	assert.equal(manager.createJobSnapshot(() => null).size, 1);
	assert.equal(manager.createJobSnapshot(() => null).size, 1);
	registration.destroy();
	assert.equal(manager.createJobSnapshot(() => null).size, 0);
}

function testUnsupportedBackendRejectsJobs() {
	const manager = new RenderTargetManager({
		supportsJobs: false,
		invalidate() {},
		async readColor() { throw new Error("unavailable"); },
	});
	const target = createTarget(manager);
	assert.throws(
		() => target.enqueueJob({ kind: "custom-pass", execute() {} }),
		/unsupported by the active backend/,
	);
}

await testAbortRequeuesAndCommitPublishesGeneration();
await testDestroyRejectsInflightTicket();
testRecurringJobsRemainRegistered();
testUnsupportedBackendRejectsJobs();
console.log("Render target manager transaction tests passed");
