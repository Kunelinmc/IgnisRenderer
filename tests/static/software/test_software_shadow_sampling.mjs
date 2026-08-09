import assert from "node:assert/strict";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	createParticleShadowVolumeGrid,
	injectParticleBatchIntoShadowVolume,
} from "../../../src/pipeline/ParticleShadowVolume.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";
import { sampleSoftwareShadow } from "../../../src/backends/software/SoftwareShadowSampler.ts";

function createFixture(size = 4, overrides = {}) {
	const definition = {
		bias: {
			constant: 0,
			slope: 0,
			texel: 0,
			max: 1,
			normal: 1,
			normalMin: 0,
		},
		sampling: {
			pcfRadius: 0,
			radius: 0,
			strength: 1,
			samples: 1,
			searchSamples: 1,
		},
		...overrides,
	};
	const shadow = { definition };
	const slice = { resolution: size, viewProjection: Matrix4.identity(), projection: Matrix4.identity(), lightDirection: { x: 0, y: -1, z: 0 } };
	const runtime = { size, depthBuffer: new Float32Array(size * size), transmissionBuffer: new Float32Array(size * size * 3), particleVolume: createParticleShadowVolumeGrid() };
	runtime.depthBuffer.fill(1);
	runtime.transmissionBuffer.fill(1);
	return { shadow, slice, runtime };
}

{
	const { shadow, slice, runtime } = createFixture();
	runtime.depthBuffer[5] = -1;
	const visibility = sampleSoftwareShadow(shadow, slice, runtime, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
	assert.ok(visibility.r < 0.01);
}

{
	const { shadow, slice, runtime } = createFixture();
	runtime.transmissionBuffer.set([0.2, 0.4, 0.6], 15);
	const visibility = sampleSoftwareShadow(shadow, slice, runtime, { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 });
	assert.ok(Math.abs(visibility.r - 0.2) < 1e-6);
	assert.ok(Math.abs(visibility.g - 0.4) < 1e-6);
	assert.ok(Math.abs(visibility.b - 0.6) < 1e-6);
}

{
	const { shadow, slice, runtime } = createFixture(4, {
		bias: {
			constant: 0,
			slope: 0,
			texel: 0,
			max: 1,
			normal: 1,
			normalMin: 1,
		},
	});
	runtime.depthBuffer[7] = -1;
	const withNormal = sampleSoftwareShadow(
		shadow,
		slice,
		runtime,
		{ x: 0, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 },
	);
	const withoutNormal = sampleSoftwareShadow(
		shadow,
		slice,
		runtime,
		{ x: 0, y: 0, z: 0 },
		null,
	);
	assert.ok(withNormal.r < 0.01);
	assert.ok(withoutNormal.r > 0.99);
}

{
	const { shadow, slice, runtime } = createFixture(4, {
		sampling: {
			pcfRadius: 0,
			radius: 2,
			strength: 1,
			samples: 1,
			searchSamples: 1,
		},
	});
	slice.projection = Matrix4.perspective(60, 1, 0.1, 10);
	runtime.depthBuffer[6] = 0;
	const visibility = sampleSoftwareShadow(
		shadow,
		slice,
		runtime,
		{ x: 0, y: 0, z: 0.5 },
		null,
	);
	assert.ok(visibility.r < 0.01);
}

{
	const { shadow, slice, runtime } = createFixture();
	const particleVolume = createParticleShadowVolumeGrid({
		width: 16,
		height: 16,
		depth: 8,
	});
	runtime.particleVolume = particleVolume;
	injectParticleBatchIntoShadowVolume(particleVolume, slice.viewProjection, {
		systemId: "particle-shadow",
		blendMode: ParticleBlendMode.Alpha,
		texture: null,
		receiveShadows: true,
		castShadows: true,
		shadowDensity: 12,
		shadowSoftness: 1,
		particles: [{
			position: { x: 0, y: 0, z: 0 },
			size: 2,
			color: { r: 255, g: 255, b: 255, a: 1 },
			rotation: 0,
			depth: 1,
			uvRect: { u0: 0, v0: 0, u1: 1, v1: 1 },
		}],
	});
	const visibility = sampleSoftwareShadow(
		shadow,
		slice,
		runtime,
		{ x: 0, y: 0, z: 0 },
		null,
	);
	assert.ok(visibility.r < 0.99);
}

console.log("software plan-native shadow sampling tests passed");
