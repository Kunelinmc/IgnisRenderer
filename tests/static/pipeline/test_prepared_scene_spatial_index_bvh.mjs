import assert from "node:assert/strict";
import { PreparedSceneTileSpatialIndex } from "../../../src/pipeline/PreparedSceneSpatialIndex.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createPacket(id) {
	return createTestDrawPacket({ id });
}

function testBVHRectQueries() {
	const opaquePackets = [
		createPacket("opaque-a"),
		createPacket("opaque-b"),
		createPacket("opaque-fallback"),
	];
	const packetRects = new Map([
		[
			"opaque-a",
			{
				x: 10,
				y: 10,
				width: 20,
				height: 20,
			},
		],
		[
			"opaque-b",
			{
				x: 120,
				y: 20,
				width: 16,
				height: 24,
			},
		],
	]);

	const spatialIndex = new PreparedSceneTileSpatialIndex({
		viewportWidth: 256,
		viewportHeight: 144,
		tileSize: 32,
		packetRects,
		opaquePackets,
		transparentPackets: [],
	});

	const firstRectHits = spatialIndex
		.queryOpaquePackets({
			x: 0,
			y: 0,
			width: 48,
			height: 48,
		})
		.map((packet) => packet.submission.id);
	assert.deepEqual(firstRectHits, ["opaque-a", "opaque-fallback"]);

	const secondRectHits = spatialIndex
		.queryOpaquePackets({
			x: 110,
			y: 0,
			width: 64,
			height: 64,
		})
		.map((packet) => packet.submission.id);
	assert.deepEqual(secondRectHits, ["opaque-b", "opaque-fallback"]);

	const emptyRectHits = spatialIndex
		.queryOpaquePackets({
			x: 200,
			y: 80,
			width: 20,
			height: 20,
		})
		.map((packet) => packet.submission.id);
	assert.deepEqual(emptyRectHits, ["opaque-fallback"]);

	const unionHits = spatialIndex
		.queryOpaquePacketsInRects([
			{
				x: 0,
				y: 0,
				width: 64,
				height: 64,
			},
			{
				x: 110,
				y: 0,
				width: 64,
				height: 64,
			},
		])
		.map((packet) => packet.submission.id);
	assert.deepEqual(unionHits, ["opaque-a", "opaque-b", "opaque-fallback"]);
}

function run() {
	testBVHRectQueries();
	console.log("Prepared scene spatial index BVH tests passed");
}

run();
