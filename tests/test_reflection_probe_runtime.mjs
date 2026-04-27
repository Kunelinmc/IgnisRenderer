import assert from "node:assert/strict";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { Texture } from "../src/core/Texture.ts";
import {
	buildReflectionProbeAtlasTexture,
	collectReflectionProbeEnvironment,
	computeParallaxCorrectedDirection,
	computeProbeDepthOcclusion,
	computeProbeRawWeight,
	samplePrefilteredEquirect,
	selectTopTwoReflectionProbes,
} from "../src/pipeline/reflectionProbeRuntime.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { CubeTexture } from "../src/core/CubeTexture.ts";

function testBlendCurveMonotonicAndContinuous() {
	const blendDistance = 0.2;
	const exponent = 1.0;
	let previous = Number.POSITIVE_INFINITY;
	for (let i = 0; i <= 40; i++) {
		const metric = 0.8 + i * 0.04;
		const weight = computeProbeRawWeight(metric, blendDistance, exponent);
		assert.ok(weight <= previous + 1e-8, "Blend weight must be monotonic");
		previous = weight;
	}

	const left = computeProbeRawWeight(0.9999, blendDistance, exponent);
	const right = computeProbeRawWeight(1.0001, blendDistance, exponent);
	assert.ok(
		Math.abs(left - right) < 1e-3,
		"Blend curve should stay continuous around the probe boundary"
	);

	const wBase = computeProbeRawWeight(1.1, blendDistance, 1.0);
	const wSharper = computeProbeRawWeight(1.1, blendDistance, 2.0);
	assert.ok(wSharper < wBase, "Higher blendExponent should sharpen the fade");
}

function testProbeDepthOcclusionAttenuatesBoundarySamples() {
	const blendDistance = 0.2;
	const deepInside = computeProbeDepthOcclusion(0.5, blendDistance);
	const nearBoundary = computeProbeDepthOcclusion(0.97, blendDistance);
	const outside = computeProbeDepthOcclusion(1.05, blendDistance);

	assert.ok(Number.isFinite(deepInside));
	assert.ok(Number.isFinite(nearBoundary));
	assert.ok(Number.isFinite(outside));
	assert.ok(deepInside > nearBoundary, "Deep samples should keep more visibility");
	assert.equal(outside, 0, "Outside-probe samples should be fully occluded");
}

function testTopTwoTieBreakByProbeId() {
	const probeA = new ReflectionProbe({ shape: "sphere", radius: 2 });
	const probeB = new ReflectionProbe({ shape: "sphere", radius: 2 });
	const probes = [probeB, probeA];
	const result = selectTopTwoReflectionProbes({ x: 0, y: 0, z: 0 }, probes);
	assert.ok(result.firstIndex >= 0);

	const selected = probes[result.firstIndex];
	const expected =
		probeA.id.localeCompare(probeB.id) < 0 ? probeA.id : probeB.id;
	assert.equal(selected.id, expected, "Tie-break should be deterministic by probe id");
	assert.ok(Math.abs(result.firstWeight + result.secondWeight - 1) < 1e-6);
}

function testParallaxIntersectionAndFallback() {
	const boxProbe = new ReflectionProbe({
		shape: "box",
		halfExtents: { x: 1, y: 1, z: 1 },
		parallaxMode: "box",
	});
	boxProbe.worldMatrix = Matrix4.identity();
	boxProbe.markRuntimeDirty();
	const box = computeParallaxCorrectedDirection(
		{ x: 0, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 },
		boxProbe
	);
	assert.equal(box.valid, true);
	assert.ok(Math.abs(box.direction.x - 1) < 1e-6);
	assert.ok(Math.abs(box.direction.y) < 1e-6);
	assert.ok(Math.abs(box.direction.z) < 1e-6);

	const sphereProbe = new ReflectionProbe({
		shape: "sphere",
		radius: 1,
		parallaxMode: "sphere",
	});
	sphereProbe.worldMatrix = Matrix4.identity();
	sphereProbe.markRuntimeDirty();
	const sphere = computeParallaxCorrectedDirection(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		sphereProbe
	);
	assert.equal(sphere.valid, true);
	assert.ok(Math.abs(sphere.direction.z - 1) < 1e-6);

	const fallback = computeParallaxCorrectedDirection(
		{ x: 2, y: 0, z: 0 },
		{ x: 1, y: 0, z: 0 },
		sphereProbe
	);
	assert.equal(fallback.valid, false);
	assert.ok(Math.abs(fallback.direction.x - 1) < 1e-6);
}

function testRuntimeCacheDirtyBehavior() {
	const originalInverse3x3 = Matrix4.inverse3x3;
	let inverseCallCount = 0;
	Matrix4.inverse3x3 = function patchedInverse(matrix) {
		inverseCallCount++;
		return originalInverse3x3(matrix);
	};

	try {
		const probe = new ReflectionProbe({
			shape: "box",
			halfExtents: { x: 1, y: 2, z: 3 },
		});
		probe.worldMatrix = Matrix4.identity();
		probe.markRuntimeDirty();

		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 1);
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 1, "Cache should not recompute without changes");

		probe.worldMatrix = Matrix4.fromTranslation([1, 0, 0]);
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 2, "Matrix change should recompute cache");

		probe.markRuntimeDirty();
		probe.getRuntimeCache();
		assert.equal(inverseCallCount, 3, "Manual dirty flag should force recompute");
	} finally {
		Matrix4.inverse3x3 = originalInverse3x3;
	}
}

function testReflectionProbeCaptureDefaultsAndClone() {
	const probe = new ReflectionProbe();
	assert.equal(probe.source, "skybox");
	assert.equal(probe.captureUpdateMode, "onSceneDirty");
	assert.equal(probe.captureIntervalSeconds, 1);
	assert.equal(probe.captureResolution.width, 512);
	assert.equal(probe.captureResolution.height, 256);
	assert.equal(probe.captureFar, 200);
	assert.equal(probe.includeSkybox, true);
	assert.equal(probe.includeMeshes, true);
	assert.equal(probe.includeTransparent, true);
	assert.equal(probe.includeParticles, true);
	assert.equal(probe.includeShadows, true);

	const cloned = probe.clone(false);
	assert.equal(cloned.source, probe.source);
	assert.equal(cloned.captureUpdateMode, probe.captureUpdateMode);
	assert.equal(cloned.captureIntervalSeconds, probe.captureIntervalSeconds);
	assert.equal(cloned.captureResolution.width, probe.captureResolution.width);
	assert.equal(cloned.captureResolution.height, probe.captureResolution.height);
	assert.equal(cloned.captureFar, probe.captureFar);
	assert.equal(cloned.includeSkybox, probe.includeSkybox);
	assert.equal(cloned.includeMeshes, probe.includeMeshes);
	assert.equal(cloned.includeTransparent, probe.includeTransparent);
	assert.equal(cloned.includeParticles, probe.includeParticles);
	assert.equal(cloned.includeShadows, probe.includeShadows);
}

function testReflectionProbeRequestCaptureFlags() {
	const probe = new ReflectionProbe({
		source: "capturedScene",
		captureUpdateMode: "manual",
	});
	assert.equal(probe.captureRequestToken, 0);
	assert.equal(probe.captureRevision, 0);

	probe.requestCapture();
	assert.equal(probe.captureRequestToken, 1);
	assert.equal(probe.captureRevision, 1);

	probe.markCaptureUpdated();
	assert.equal(probe.captureRevision, 2);
}

function testCubemapSpecularSamplingAndAtlasBuild() {
	const cubeA = createTinyCubeTexture([
		new Uint8Array([255, 0, 0, 255]),
		new Uint8Array([0, 255, 0, 255]),
		new Uint8Array([0, 0, 255, 255]),
		new Uint8Array([255, 255, 0, 255]),
		new Uint8Array([255, 0, 255, 255]),
		new Uint8Array([0, 255, 255, 255]),
	]);
	const plusX = samplePrefilteredEquirect(cubeA, { x: 1, y: 0, z: 0 }, 0);
	assert.ok(plusX.r > 0.9 && plusX.g < 0.1 && plusX.b < 0.1);
	const plusZ = samplePrefilteredEquirect(cubeA, { x: 0, y: 0, z: 1 }, 0);
	assert.ok(plusZ.r > 0.9 && plusZ.g < 0.1 && plusZ.b > 0.9);

	const cubeB = createTinyCubeTexture([
		new Uint8Array([64, 64, 64, 255]),
		new Uint8Array([64, 64, 64, 255]),
		new Uint8Array([64, 64, 64, 255]),
		new Uint8Array([64, 64, 64, 255]),
		new Uint8Array([64, 64, 64, 255]),
		new Uint8Array([64, 64, 64, 255]),
	]);
	const probeA = new ReflectionProbe({ prefilteredMap: cubeA });
	const probeB = new ReflectionProbe({ prefilteredMap: cubeB });
	const atlas = buildReflectionProbeAtlasTexture([probeA, probeB]);
	assert.ok(atlas);
	assert.equal(atlas.width, 8);
	assert.equal(atlas.height, 2);
	assert.ok(atlas.data || atlas.mipmaps.length > 0);
}

function testAtlasCacheInvalidatesWhenProbeTextureObjectChanges() {
	const probe = new ReflectionProbe();
	const firstMap = createTinyEquirectTexture(0.2);
	const secondMap = createTinyEquirectTexture(0.8);
	probe.prefilteredMap = firstMap;

	const firstAtlas = buildReflectionProbeAtlasTexture([probe]);
	assert.ok(firstAtlas);
	const firstValue = readTextureFirstChannel(firstAtlas);
	assert.ok(Math.abs(firstValue - 0.2) < 1e-6);

	probe.prefilteredMap = secondMap;
	const secondAtlas = buildReflectionProbeAtlasTexture([probe]);
	assert.ok(secondAtlas);
	const secondValue = readTextureFirstChannel(secondAtlas);
	assert.ok(Math.abs(secondValue - 0.8) < 1e-6);
	assert.notEqual(secondAtlas, firstAtlas);
}

function testCollectReflectionProbeEnvironmentKeepsAtlasWhenProbeFormatsDiffer() {
	const lowQualityMap = createEquirectTexture(0.2, 1, 1, 1);
	const highQualityMap = createEquirectTexture(0.8, 8, 4, 4);
	const lowProbe = new ReflectionProbe({ prefilteredMap: lowQualityMap });
	const highProbe = new ReflectionProbe({ prefilteredMap: highQualityMap });

	const collected = collectReflectionProbeEnvironment([lowProbe, highProbe], 8);
	assert.equal(collected.probes.length, 1);
	assert.equal(collected.probes[0].id, highProbe.id);
	assert.ok(collected.atlas);
	assert.equal(collected.atlas.width, 8);
	assert.equal(collected.atlas.height, 4);
}

function createTinyCubeTexture(faces) {
	return new CubeTexture({
		faces,
		size: 1,
		colorSpace: "sRGB",
	});
}

function createTinyEquirectTexture(value) {
	const data = new Float32Array([value, value, value, 1]);
	const texture = new Texture(data, 1, 1, "HDR");
	texture.mipmaps = [new Float32Array(data)];
	return texture;
}

function createEquirectTexture(value, width, height, mipCount) {
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const resolvedMipCount = Math.max(1, Math.floor(mipCount));
	const mipmaps = [];
	for (let level = 0; level < resolvedMipCount; level++) {
		const mipWidth = Math.max(1, resolvedWidth >> level);
		const mipHeight = Math.max(1, resolvedHeight >> level);
		const mipData = new Float32Array(mipWidth * mipHeight * 4);
		for (let i = 0; i < mipData.length; i += 4) {
			mipData[i] = value;
			mipData[i + 1] = value;
			mipData[i + 2] = value;
			mipData[i + 3] = 1;
		}
		mipmaps.push(mipData);
	}
	const texture = new Texture(mipmaps[0], resolvedWidth, resolvedHeight, "HDR");
	texture.mipmaps = mipmaps;
	texture.data = mipmaps[0];
	return texture;
}

function readTextureFirstChannel(texture) {
	const data = texture.data ?? texture.mipmaps[0];
	return data ? data[0] : 0;
}

function run() {
	testBlendCurveMonotonicAndContinuous();
	testProbeDepthOcclusionAttenuatesBoundarySamples();
	testTopTwoTieBreakByProbeId();
	testParallaxIntersectionAndFallback();
	testRuntimeCacheDirtyBehavior();
	testReflectionProbeCaptureDefaultsAndClone();
	testReflectionProbeRequestCaptureFlags();
	testCubemapSpecularSamplingAndAtlasBuild();
	testAtlasCacheInvalidatesWhenProbeTextureObjectChanges();
	testCollectReflectionProbeEnvironmentKeepsAtlasWhenProbeFormatsDiffer();
	console.log("Reflection probe runtime tests passed");
}

run();
