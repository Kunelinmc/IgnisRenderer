import assert from "node:assert/strict";
import { bindWebGLGlobalUniforms } from "../src/renderers/webgl/WebGLGlobalUniformBinder.ts";
import { WEBGL_MAX_POINT_LIGHTS } from "../src/renderers/webgl/constants.ts";

function approxEqual(actual, expected, epsilon = 1e-5) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${actual} ~= ${expected}`
	);
}

function createFakeGLRecorder() {
	const uniform1iCalls = new Map();
	const uniform4fvCalls = new Map();
	return {
		TEXTURE0: 0x84c0,
		uniform1i(location, value) {
			uniform1iCalls.set(location, value);
		},
		uniform4fv(location, values) {
			uniform4fvCalls.set(location, values);
		},
		activeTexture() {},
		_calls: {
			uniform1iCalls,
			uniform4fvCalls,
		},
	};
}

function createPointLights(count) {
	return Array.from({ length: count }, (_, index) => ({
		position: [index + 1, index + 2, index + 3],
		range: 10 + index,
		color: [0.1 * (index + 1), 0.2 * (index + 1), 0.3 * (index + 1)],
	}));
}

function createBinderHost(gl, pointLights) {
	return {
		_gl: gl,
		_lightState: {
			ambientColor: [0, 0, 0],
			directionalLights: [],
			directionalShadows: [],
			pointLights,
			spotLights: [],
			spotShadows: [],
			clusteredLights: [],
			envSpecularMap: null,
			localLightProbeCount: 0,
			localLightProbes: [],
			reflectionProbeCount: 0,
			reflectionProbes: [],
		},
		_textures: {
			getEnvironmentSpecularTexture() {
				return { texture: null, isLinear: false };
			},
			getBRDFLUTTexture() {
				return { texture: null, isLinear: false };
			},
		},
		_clusteredLighting: {
			getState() {
				return {
					enabled: false,
					screenWidth: 1,
					screenHeight: 1,
					tilesX: 1,
					tilesY: 1,
					zSlices: 1,
					maxLightsPerCluster: 1,
					logScale: 0,
					logBias: 0,
					headerTexture: null,
					headerTexWidth: 1,
					headerTexHeight: 1,
					indexTexture: null,
					indexTexWidth: 1,
					indexTexHeight: 1,
					lightTexture: null,
					lightTexWidth: 1,
					lightTexHeight: 1,
				};
			},
		},
		_shadowAtlasTexture: null,
		_shadowAtlasTileSize: 0,
		_taaJitter: new Float32Array([0, 0, 0, 0]),
		_prevViewProjection: null,
		_shAmbientTexture: null,
		_shAmbientTextureWidth: 1,
		_shAmbientTextureHeight: 1,
		_localLightProbeSHTexture: null,
		_localLightProbeSHTextureWidth: 1,
		_localLightProbeSHTextureHeight: 1,
		_fogParams0: new Float32Array([0, 0, 0, 0]),
		_fogParams1: new Float32Array([0, 0, 0, 0]),
		_updateFogParams() {},
		_uploadSHAmbientCoefficients() {
			return false;
		},
		_uploadLocalLightProbeCoefficients() {
			return false;
		},
	};
}

function createFrameContext() {
	return {
		camera: {
			getWorldPosition() {
				return { x: 0, y: 0, z: 0 };
			},
		},
		features: {
			enableFog: false,
			enableClusteredLighting: false,
			enableLighting: true,
			enableShadows: false,
			enableSH: false,
		},
		shAmbientCoeffs: null,
	};
}

function testPointLightUniformPackingUsesConfiguredBudget() {
	const gl = createFakeGLRecorder();
	const pointLights = createPointLights(6);
	const host = createBinderHost(gl, pointLights);
	const frameContext = createFrameContext();

	const pointLightCount = { id: "point-count" };
	const pointLightPositionRange = { id: "point-position-range" };
	const pointLightColor = { id: "point-color" };
	const sceneProgram = {
		uniforms: {
			pointLightCount,
			pointLightPositionRange,
			pointLightColor,
		},
	};

	bindWebGLGlobalUniforms(host, sceneProgram, frameContext);

	assert.equal(gl._calls.uniform1iCalls.get(pointLightCount), 6);

	const packedPosition = gl._calls.uniform4fvCalls.get(pointLightPositionRange);
	assert.equal(packedPosition.length, WEBGL_MAX_POINT_LIGHTS * 4);
	assert.equal(packedPosition[20], 6);
	assert.equal(packedPosition[21], 7);
	assert.equal(packedPosition[22], 8);
	assert.equal(packedPosition[23], 15);
	assert.equal(packedPosition[24], 0);

	const packedColor = gl._calls.uniform4fvCalls.get(pointLightColor);
	assert.equal(packedColor.length, WEBGL_MAX_POINT_LIGHTS * 4);
	approxEqual(packedColor[20], 0.6);
	approxEqual(packedColor[21], 1.2);
	approxEqual(packedColor[22], 1.8);
	assert.equal(packedColor[23], 0);
	assert.equal(packedColor[24], 0);
}

function run() {
	testPointLightUniformPackingUsesConfiguredBudget();
	console.log("WebGL global uniform binder light packing tests passed");
}

run();
