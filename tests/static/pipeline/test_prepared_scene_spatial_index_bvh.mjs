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

function testDeferredTreesKeepConstructionSnapshot() {
	const original = PreparedSceneTileSpatialIndex.prototype._buildPacketTree;
	let builds = 0;
	PreparedSceneTileSpatialIndex.prototype._buildPacketTree = function (...args) {
		builds++;
		return original.apply(this, args);
	};
	try {
		const opaque = [createPacket("opaque"), createPacket("missing")];
		const transparent = [createPacket("transparent"), createPacket("outside")];
		const rect = { x: 10.2, y: 10.2, width: 10, height: 10 };
		const packetRects = new Map([
			["opaque", rect], ["transparent", { ...rect }],
			["outside", { x: -20, y: -20, width: 1, height: 1 }],
		]);
		const index = new PreparedSceneTileSpatialIndex({
			viewportWidth: 64, viewportHeight: 64, tileSize: 32,
			opaquePackets: opaque, transparentPackets: transparent, packetRects,
		});
		assert.equal(builds, 0, "construction snapshots entries without building either tree");
		const opaqueSnapshot = opaque.slice();
		const transparentSnapshot = transparent.slice();
		opaque.reverse();
		opaque.push(createPacket("late"));
		transparent.length = 0;
		rect.x = 50;
		packetRects.clear();
		const query = { x: 10, y: 10, width: 11, height: 11 };
		assert.deepEqual(index.queryOpaquePackets(query), opaqueSnapshot);
		assert.equal(builds, 1, "only opaque queries initialize the opaque tree");
		assert.deepEqual(index.queryOpaquePacketsInRects([query, query]), opaqueSnapshot);
		assert.deepEqual(index.queryOpaquePacketsInRects([]), [opaqueSnapshot[1]]);
		assert.equal(builds, 1, "repeated queries reuse the tree");
		assert.deepEqual(index.queryTransparentPacketsInRects([query, query]), transparentSnapshot);
		assert.equal(builds, 2);
		assert.deepEqual(index.queryTransparentPackets(query), transparentSnapshot);
		assert.deepEqual(index.queryTransparentPacketsInRects([]), [transparentSnapshot[1]]);
		assert.deepEqual(index.queryOpaquePackets({ x: 50, y: 50, width: 2, height: 2 }), [opaqueSnapshot[1]]);
		assert.equal(builds, 2);

		const empty = new PreparedSceneTileSpatialIndex({
			viewportWidth: 64, viewportHeight: 64, tileSize: 32,
			opaquePackets: [], transparentPackets: [], packetRects: new Map(),
		});
		const beforeEmpty = builds;
		assert.deepEqual(empty.queryOpaquePackets(query), []);
		const afterOpaque = builds;
		assert.ok(afterOpaque - beforeEmpty <= 1);
		assert.deepEqual(empty.queryOpaquePacketsInRects([query]), []);
		assert.equal(builds, afterOpaque, "an empty tree must not reinitialize");
		assert.deepEqual(empty.queryTransparentPacketsInRects([]), []);
		const afterTransparent = builds;
		assert.ok(afterTransparent - afterOpaque <= 1);
		assert.deepEqual(empty.queryTransparentPackets(query), []);
		assert.equal(builds, afterTransparent);
	} finally {
		PreparedSceneTileSpatialIndex.prototype._buildPacketTree = original;
	}
}

function testBVHQueriesMatchLinearSnapshot() {
	let seed = 20260929;
	const random = () => {
		seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
		return seed / 0x100000000;
	};
	const packets = Array.from({ length: 200 }, (_, i) => createPacket(`packet-${i}`));
	const rects = new Map(packets.filter((_, i) => i % 17 !== 0).map((packet) => [
		packet.submission.id,
		{ x: random() * 220, y: random() * 110, width: 1 + random() * 30, height: 1 + random() * 30 },
	]));
	const index = new PreparedSceneTileSpatialIndex({
		viewportWidth: 256, viewportHeight: 144, tileSize: 32,
		opaquePackets: packets, transparentPackets: packets, packetRects: rects,
	});
	for (let i = 0; i < 30; i++) {
		const query = { x: Math.floor(random() * 200), y: Math.floor(random() * 100), width: 40, height: 30 };
		const expected = packets.filter((packet) => {
			const rect = rects.get(packet.submission.id);
			return !rect || (
				Math.floor(rect.x) <= query.x + query.width &&
				Math.ceil(rect.x + rect.width) >= query.x &&
				Math.floor(rect.y) <= query.y + query.height &&
				Math.ceil(rect.y + rect.height) >= query.y
			);
		});
		assert.deepEqual(index.queryOpaquePackets(query), expected);
		assert.deepEqual(index.queryTransparentPackets(query), expected);
	}
}

function run() {
	testDeferredTreesKeepConstructionSnapshot();
	testBVHQueriesMatchLinearSnapshot();
	testBVHRectQueries();
	console.log("Prepared scene spatial index BVH tests passed");
}

run();
