import assert from "node:assert/strict";
import { bindWebGLGlobalUniforms } from "../../../src/renderers/webgl/WebGLGlobalUniformBinder.ts";
import { MAX_POINT_LIGHTS } from "../../../src/renderers/constants.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function approxEqual(actual, expected, epsilon = 1e-5) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${actual} ~= ${expected}`
	);
}

function createFakeGLRecorder() {
	const uniform1iCalls = new Map();
	const uniform3fvCalls = new Map();
	const uniform4fvCalls = new Map();
	return {
		TEXTURE0: 0x84c0,
		TEXTURE_2D: 0x0de1,
		uniform1i(location, value) {
			uniform1iCalls.set(location, value);
		},
		uniform3fv(location, values) {
			uniform3fvCalls.set(location, values.slice());
		},
		uniform4fv(location, values) {
			uniform4fvCalls.set(location, values);
		},
		activeTexture() {},
		bindTexture() {},
		_calls: {
			uniform1iCalls,
			uniform3fvCalls,
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
			enableClusteredLighting: false,
			enableLighting: true,
			enableShadows: false,
			enableSH: false,
		},
		postProcess: createResolvedPostProcess({
			fog: { enabled: false },
		}),
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
	assert.equal(packedPosition.length, MAX_POINT_LIGHTS * 4);
	assert.equal(packedPosition[20], 6);
	assert.equal(packedPosition[21], 7);
	assert.equal(packedPosition[22], 8);
	assert.equal(packedPosition[23], 15);
	assert.equal(packedPosition[24], 0);

	const packedColor = gl._calls.uniform4fvCalls.get(pointLightColor);
	assert.equal(packedColor.length, MAX_POINT_LIGHTS * 4);
	approxEqual(packedColor[20], 0.6);
	approxEqual(packedColor[21], 1.2);
	approxEqual(packedColor[22], 1.8);
	assert.equal(packedColor[23], 0);
	assert.equal(packedColor[24], 0);
}

function testSHAmbientCoefficientsUseUniformVectors() {
	const gl = createFakeGLRecorder();
	const host = createBinderHost(gl, []);
	const frameContext = createFrameContext();
	const shAmbientCoeffs = Array.from({ length: 16 }, (_, index) => ({
		r: index + 1,
		g: index + 2,
		b: index + 3,
	}));
	frameContext.features.enableSH = true;
	frameContext.shAmbientCoeffs = shAmbientCoeffs;

	const enableSH = { id: "enable-sh" };
	const shAmbientCoeffsLocation = { id: "sh-ambient-coeffs" };
	const sceneProgram = {
		uniforms: {
			enableSH,
			shAmbientCoeffs: shAmbientCoeffsLocation,
		},
	};

	bindWebGLGlobalUniforms(host, sceneProgram, frameContext);

	assert.equal(gl._calls.uniform1iCalls.get(enableSH), 1);
	const packed = gl._calls.uniform3fvCalls.get(shAmbientCoeffsLocation);
	assert.equal(packed.length, 48);
	assert.equal(packed[0], 1);
	assert.equal(packed[1], 2);
	assert.equal(packed[2], 3);
	assert.equal(packed[45], 16);
	assert.equal(packed[46], 17);
	assert.equal(packed[47], 18);
}

function testLocalLightProbeSHUsesFreedTextureUnit() {
	const gl = createFakeGLRecorder();
	const host = createBinderHost(gl, []);
	const frameContext = createFrameContext();
	const localLightProbeCoeffs = { id: "local-light-probe-coeffs" };
	const sceneProgram = {
		uniforms: {
			localLightProbeCoeffs,
		},
	};

	bindWebGLGlobalUniforms(host, sceneProgram, frameContext);

	assert.equal(gl._calls.uniform1iCalls.get(localLightProbeCoeffs), 4);
}

function run() {
	testPointLightUniformPackingUsesConfiguredBudget();
	testSHAmbientCoefficientsUseUniformVectors();
	testLocalLightProbeSHUsesFreedTextureUnit();
	console.log("WebGL global uniform binder light packing tests passed");
}

run();
