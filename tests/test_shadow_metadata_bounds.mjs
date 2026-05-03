import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { resolveShadowCasterBounds } from "../src/pipeline/ShadowMetadata.ts";
import {
	mergeParticleShadowBounds,
	resolveParticleShadowCasterBounds,
} from "../src/pipeline/ParticleShadowVolume.ts";
import { ParticleSystem } from "../src/particles/ParticleSystem.ts";
import { ParticleBlendMode } from "../src/particles/types.ts";

function createPacket(id, center, radius = 1) {
	return {
		id,
		worldBounds: {
			center,
			radius,
		},
	};
}

function testShadowBoundsUseCameraVisibleCasters() {
	const camera = new Camera();
	camera.position.set(0, 0, 0);
	camera.updateMatrices();

	const visibleCaster = createPacket("visible", { x: 0, y: 0, z: -10 }, 1);
	const farDroppedCaster = createPacket(
		"farDropped",
		{ x: 0, y: -800, z: -10 },
		2
	);

	const bounds = resolveShadowCasterBounds(
		[visibleCaster, farDroppedCaster],
		{ center: { x: 0, y: -400, z: -10 }, radius: 1000 },
		camera
	);

	assert.ok(Math.abs(bounds.center.y) < 1e-6);
	assert.ok(Math.abs(bounds.center.z + 10) < 1e-6);
	assert.ok(
		bounds.radius < 10,
		`Expected tight visible caster bounds, got radius=${bounds.radius}`
	);
}

function testShadowBoundsFallbackToNearestCasterWhenNothingVisible() {
	const camera = new Camera();
	camera.position.set(0, 0, 0);
	camera.updateMatrices();

	const nearBehindCaster = createPacket("nearBehind", { x: 0, y: 0, z: 8 }, 1);
	const farBehindCaster = createPacket("farBehind", { x: 0, y: 0, z: 320 }, 1);

	const bounds = resolveShadowCasterBounds(
		[nearBehindCaster, farBehindCaster],
		{ center: { x: 0, y: 0, z: 1000 }, radius: 4000 },
		camera
	);

	assert.ok(
		Math.abs(bounds.center.z - 8) < 1e-6,
		`Expected nearest caster center.z=8, got ${bounds.center.z}`
	);
	assert.ok(
		bounds.radius < 10,
		`Expected nearest caster radius to stay tight, got ${bounds.radius}`
	);
}

function testParticleCastersExpandShadowBounds() {
	const particle = new ParticleSystem({
		position: { x: 12, y: 0, z: 0 },
		blendMode: ParticleBlendMode.Alpha,
		castShadows: true,
		shadowDensity: 1,
		emit: {
			spawnRadius: 1,
			lifetimeRange: [1, 1],
			speedRange: [0, 0],
			sizeRange: [2, 2],
		},
	});
	const additive = new ParticleSystem({
		position: { x: 100, y: 0, z: 0 },
		blendMode: ParticleBlendMode.Additive,
		castShadows: true,
		shadowDensity: 1,
	});

	const particleBounds = resolveParticleShadowCasterBounds([particle, additive]);
	assert.ok(particleBounds);
	assert.ok(particleBounds.center.x > 10);
	assert.ok(particleBounds.radius > 0);

	const merged = mergeParticleShadowBounds(
		{ center: { x: 0, y: 0, z: 0 }, radius: 1 },
		particleBounds
	);
	assert.ok(
		merged.radius > 6,
		`Expected particle caster to expand shadow bounds, got ${merged.radius}`
	);
}

function run() {
	testShadowBoundsUseCameraVisibleCasters();
	testShadowBoundsFallbackToNearestCasterWhenNothingVisible();
	testParticleCastersExpandShadowBounds();
	console.log("Shadow metadata bounds tests passed");
}

run();
