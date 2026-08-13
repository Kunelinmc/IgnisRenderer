import assert from "node:assert/strict";

import { WebGPUFrameMessageDispatchError } from "../../../src/foundation/Error.ts";
import {
	defineWebGPUFrameMessage,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessage.ts";
import {
	WebGPUFrameMessageRegistry,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessageRegistry.ts";

const seed = (id, phase = "analysis") => defineWebGPUFrameMessage({
	id,
	ownerId: "core",
	phase,
	seeded: true,
});
const one = (id, ownerId, phase = "analysis") => defineWebGPUFrameMessage({
	id,
	ownerId,
	phase,
});
const many = (id, ownerId, phase = "analysis") => defineWebGPUFrameMessage({
	id,
	ownerId,
	phase,
	cardinality: "many",
});

async function testDeterministicWavesAndDeclaredAccess() {
	const input = seed("test:input");
	const middleA = one("test:middle-a", "a");
	const middleB = one("test:middle-b", "b");
	const output = one("test:output", "consumer");
	const ordering = [];
	const registry = new WebGPUFrameMessageRegistry();
	registry.register({
		id: "produce",
		moduleId: "b",
		phase: "analysis",
		inputs: [{ descriptor: input }],
		outputs: [middleB],
		async run(messages, publisher) {
			await Promise.resolve();
			ordering.push("b");
			publisher.publish(middleB, messages.get(input) + "b");
		},
	});
	registry.register({
		id: "produce",
		moduleId: "a",
		phase: "analysis",
		inputs: [{ descriptor: input }],
		outputs: [middleA],
		async run(messages, publisher) {
			await new Promise((resolve) => setTimeout(resolve, 1));
			ordering.push("a");
			publisher.publish(middleA, messages.get(input) + "a");
			assert.throws(() => messages.get(output), /undeclared input/);
		},
	});
	registry.register({
		id: "consume",
		moduleId: "consumer",
		phase: "analysis",
		inputs: [{ descriptor: middleA }, { descriptor: middleB }],
		outputs: [output],
		run(messages, publisher) {
			publisher.publish(output, messages.get(middleA) + messages.get(middleB));
		},
	});
	registry.seal();
	const snapshot = await registry.dispatch("analysis", {
		seeds: [{ descriptor: input, value: "root-" }],
	});
	assert.deepEqual(ordering.sort(), ["a", "b"]);
	assert.equal(snapshot.get(output), "root-aroot-b");
}

function testSealValidation() {
	const missing = one("test:missing", "missing");
	const output = one("test:unused", "consumer");
	const missingRegistry = new WebGPUFrameMessageRegistry();
	missingRegistry.register({
		id: "consumer",
		moduleId: "consumer",
		phase: "analysis",
		inputs: [{ descriptor: missing }],
		outputs: [output],
		run() {},
	});
	assert.throws(() => missingRegistry.seal(), /has no producer/);

	const duplicate = many("test:duplicate", "shared");
	const single = one("test:single", "shared");
	const duplicateRegistry = new WebGPUFrameMessageRegistry();
	for (const moduleId of ["a", "b"]) {
		duplicateRegistry.register({
			id: "producer",
			moduleId,
			phase: "analysis",
			outputs: [single, duplicate],
			run(_messages, publisher) {
				publisher.publish(single, moduleId);
				publisher.publish(duplicate, moduleId);
			},
		});
	}
	assert.throws(() => duplicateRegistry.seal(), /duplicate single-value producers/);

	const a = one("test:cycle-a", "a");
	const b = one("test:cycle-b", "b");
	const cycleRegistry = new WebGPUFrameMessageRegistry();
	cycleRegistry.register({
		id: "a",
		moduleId: "a",
		phase: "analysis",
		inputs: [{ descriptor: b }],
		outputs: [a],
		run() {},
	});
	cycleRegistry.register({
		id: "b",
		moduleId: "b",
		phase: "analysis",
		inputs: [{ descriptor: a }],
		outputs: [b],
		run() {},
	});
	assert.throws(() => cycleRegistry.seal(), /contains a cycle/);

	const analysisOutput = one("test:analysis-output", "analysis", "analysis");
	const configurationOutput = one(
		"test:configuration-output",
		"configuration",
		"configuration",
	);
	const backwardRegistry = new WebGPUFrameMessageRegistry();
	backwardRegistry.register({
		id: "configuration",
		moduleId: "configuration",
		phase: "configuration",
		outputs: [configurationOutput],
		run(_messages, publisher) {
			publisher.publish(configurationOutput, true);
		},
	});
	backwardRegistry.register({
		id: "analysis",
		moduleId: "analysis",
		phase: "analysis",
		inputs: [{ descriptor: configurationOutput }],
		outputs: [analysisOutput],
		run(_messages, publisher) {
			publisher.publish(analysisOutput, true);
		},
	});
	assert.throws(() => backwardRegistry.seal(), /depends on later phase message/);
}

async function testPublicationContracts() {
	const declared = one("test:declared-output", "publisher");
	const foreign = seed("test:foreign-output");
	const undeclaredRegistry = new WebGPUFrameMessageRegistry();
	undeclaredRegistry.register({
		id: "publisher",
		moduleId: "publisher",
		phase: "analysis",
		outputs: [declared],
		run(_messages, publisher) {
			publisher.publish(foreign, true);
		},
	});
	undeclaredRegistry.seal();
	await assert.rejects(
		undeclaredRegistry.dispatch("analysis"),
		(error) => /cannot publish undeclared output/.test(error.cause?.message),
	);

	const missingRegistry = new WebGPUFrameMessageRegistry();
	missingRegistry.register({
		id: "publisher",
		moduleId: "publisher",
		phase: "analysis",
		outputs: [declared],
		run() {},
	});
	missingRegistry.seal();
	await assert.rejects(
		missingRegistry.dispatch("analysis"),
		(error) => /must publish .* exactly once/.test(error.cause?.message),
	);
}

async function testTransactionalFailure() {
	const output = one("test:failure-output", "failure");
	const registry = new WebGPUFrameMessageRegistry();
	registry.register({
		id: "failure",
		moduleId: "failure",
		phase: "analysis",
		outputs: [output],
		async run() {
			await Promise.resolve();
			throw new Error("expected failure");
		},
	});
	registry.seal();
	await assert.rejects(
		registry.dispatch("analysis"),
		(error) => {
			assert.ok(error instanceof WebGPUFrameMessageDispatchError);
			assert.equal(error.phase, "analysis");
			assert.equal(error.moduleId, "failure");
			assert.match(error.cause.message, /expected failure/);
			return true;
		},
	);
}

await testDeterministicWavesAndDeclaredAccess();
testSealValidation();
await testPublicationContracts();
await testTransactionalFailure();
console.log("WebGPU frame message registry tests passed");
