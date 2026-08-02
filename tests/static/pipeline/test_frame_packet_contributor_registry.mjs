import assert from "node:assert/strict";

import {
	FramePacketContributorRegistry,
} from "../../../src/pipeline/FramePacketContributorRegistry.ts";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	createTransientStore,
} from "../../../src/pipeline/types.ts";

function createPacket(id, passFlags = 0) {
	return { id, passFlags };
}

function createContext({ transient = createTransientStore(), camera = {}, scene } = {}) {
	return {
		transient,
		viewCamera: camera,
		scene: scene ?? {
			opaquePackets: [createPacket("scene-opaque")],
			transparentPackets: [
				createPacket("scene-transparent", DRAW_PACKET_FLAG_TRANSPARENT),
			],
			shadowCasterPackets: [
				createPacket("scene-shadow", DRAW_PACKET_FLAG_SHADOW_CASTER),
			],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
	};
}

function testCompositionCachingAndViewIsolation() {
	const opaque = createPacket("contributor-opaque", DRAW_PACKET_FLAG_SHADOW_CASTER);
	const transparent = createPacket(
		"contributor-transparent",
		DRAW_PACKET_FLAG_TRANSPARENT |
			DRAW_PACKET_FLAG_SHADOW_TRANSMITTER |
			DRAW_PACKET_FLAG_REFLECTIVE,
	);
	const registry = new FramePacketContributorRegistry();
	registry.register({
		id: "test-contributor",
		supports: () => true,
		contribute: (_context, sink) => {
			sink.add(opaque);
			sink.add(transparent);
		},
	});
	const context = createContext();
	const packets = registry.prepare(context, "main");

	assert.strictEqual(registry.prepare(context, "main"), packets);
	assert.deepEqual(packets.all.map((packet) => packet.id), [
		"scene-opaque",
		"scene-transparent",
		"contributor-opaque",
		"contributor-transparent",
	]);
	assert.deepEqual(packets.opaque.map((packet) => packet.id), [
		"scene-opaque",
		"contributor-opaque",
	]);
	assert.deepEqual(packets.transparent.map((packet) => packet.id), [
		"scene-transparent",
		"contributor-transparent",
	]);
	assert.strictEqual(packets.opaque[1], opaque);
	assert.strictEqual(packets.transparent[1], transparent);
	assert.deepEqual(packets.shadowCasters.map((packet) => packet.id), [
		"scene-shadow",
		"contributor-opaque",
	]);
	assert.deepEqual(packets.shadowTransmitters.map((packet) => packet.id), [
		"contributor-transparent",
	]);
	assert.deepEqual(packets.reflective.map((packet) => packet.id), [
		"contributor-transparent",
	]);

	const captureContext = createContext({
		transient: createTransientStore(context.transient),
		camera: {},
	});
	const capturePackets = registry.prepare(captureContext, "probe-capture");
	assert.notStrictEqual(capturePackets, packets);
	assert.strictEqual(registry.prepare(captureContext, "probe-capture"), capturePackets);
	assert.notStrictEqual(registry.prepare(context, "planar-reflection"), packets);
}

function testRegistrationAndFailureRules() {
	const registry = new FramePacketContributorRegistry();
	registry.register({ id: "once", supports: () => false, contribute() {} });
	assert.throws(
		() => registry.register({ id: "once", supports: () => false, contribute() {} }),
		/already registered/,
	);
	registry.prepare(createContext(), "main");
	assert.throws(
		() => registry.register({ id: "later", supports: () => false, contribute() {} }),
		/cannot be registered/,
	);

	let attempts = 0;
	const failing = new FramePacketContributorRegistry();
	failing.register({
		id: "failing",
		supports: () => true,
		contribute() {
			attempts++;
			throw new Error("expected contributor failure");
		},
	});
	const context = createContext();
	assert.throws(() => failing.prepare(context, "main"), /expected contributor failure/);
	assert.throws(() => failing.prepare(context, "main"), /expected contributor failure/);
	assert.equal(attempts, 2);
}

testCompositionCachingAndViewIsolation();
testRegistrationAndFailureRules();
console.log("Frame packet contributor registry tests passed");
