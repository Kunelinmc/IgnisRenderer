import assert from "node:assert/strict";
import { CameraType } from "../../../src/cameras/Camera.ts";
import { Logger } from "../../../src/foundation/Logger.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { WebGLClusteredLightingRuntime } from "../../../src/renderers/webgl/WebGLClusteredLightingRuntime.ts";

function createFakeGL() {
	let textureId = 0;
	let boundTexture = null;
	const uploads = new Map();
	return {
		TEXTURE_2D: 0x0de1,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		CLAMP_TO_EDGE: 0x812f,
		NEAREST: 0x2600,
		RGBA: 0x1908,
		FLOAT: 0x1406,
		RGBA32F: 0x8814,
		createTexture() {
			return { id: `tex-${++textureId}` };
		},
		deleteTexture() {},
		bindTexture(_target, texture) {
			boundTexture = texture;
		},
		texParameteri() {},
		texImage2D(
			_target,
			_level,
			_internalFormat,
			width,
			height,
			_border,
			_format,
			_type,
			pixels
		) {
			uploads.set(boundTexture, {
				width,
				height,
				pixels: new Float32Array(pixels),
			});
		},
		getUpload(texture) {
			return uploads.get(texture);
		},
	};
}

function createPerspectiveContext(overrides = {}) {
	const { features: featureOverrides = {}, ...rest } = overrides;
	return {
		attachments: {
			width: 640,
			height: 360,
		},
		camera: {
			type: CameraType.Perspective,
			near: 0.1,
			far: 200,
			fov: 60,
			aspectRatio: 640 / 360,
			viewMatrix: Matrix4.identity(),
			viewProjectionMatrix: Matrix4.perspective(60, 640 / 360, 0.1, 200),
		},
		features: {
			enableLighting: true,
			enableClusteredLighting: true,
			clusteredLightingOptions: {
				tileSizePx: 64,
				zSlices: 8,
				maxLights: 16,
				maxLightsPerCluster: 8,
			},
			...featureOverrides,
		},
		...rest,
	};
}

function createLightState(overrides = {}) {
	return {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		clusteredLights: [
			{
				type: 0,
				position: [0, 0, -6],
				range: 5,
				direction: [0, 0, 0],
				outerCos: -2,
				innerCos: -2,
				color: [1, 1, 1],
				castsShadow: false,
				shadowIndex: 0,
			},
		],
		envSpecularMap: null,
		reflectionProbeCount: 0,
		reflectionProbes: [],
		...overrides,
	};
}

function withCapturedWarnings(action) {
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: {
			warn: (...args) => {
				warnings.push(args.map((part) => String(part)).join(" "));
			},
		},
		resetOnceKeys: true,
	});
	try {
		action(warnings);
	} finally {
		Logger.reset();
	}
}

function testPerspectiveBuildsClusterTextures() {
	withCapturedWarnings((warnings) => {
		const runtime = new WebGLClusteredLightingRuntime(createFakeGL());
		const context = createPerspectiveContext();
		const lights = createLightState();

		runtime.prepare(context, lights, 4096);
		const state = runtime.getState();

		assert.equal(state.enabled, true);
		assert.ok(state.headerTexture);
		assert.ok(state.indexTexture);
		assert.ok(state.lightTexture);
		assert.equal(warnings.length, 0);
	});
}

function testNonPerspectiveFallsBackWithWarning() {
	withCapturedWarnings((warnings) => {
		const runtime = new WebGLClusteredLightingRuntime(createFakeGL());
		const context = createPerspectiveContext({
			camera: {
				type: CameraType.Orthographic,
				near: 0.1,
				far: 200,
				fov: 60,
				aspectRatio: 1,
				viewMatrix: Matrix4.identity(),
				viewProjectionMatrix: Matrix4.identity(),
			},
		});
		const lights = createLightState();

		runtime.prepare(context, lights, 4096);
		const state = runtime.getState();

		assert.equal(state.enabled, false);
		assert.ok(
			warnings.some((warning) =>
				warning.includes("only supports perspective cameras")
			)
		);
	});
}

function testLightBudgetWarning() {
	withCapturedWarnings((warnings) => {
		const runtime = new WebGLClusteredLightingRuntime(createFakeGL());
		const context = createPerspectiveContext({
			features: {
				clusteredLightingOptions: {
					tileSizePx: 64,
					zSlices: 8,
					maxLights: 1,
					maxLightsPerCluster: 8,
				},
			},
		});
		const lights = createLightState({
			clusteredLights: [
				{
					type: 0,
					position: [0, 0, -6],
					range: 5,
					direction: [0, 0, 0],
					outerCos: -2,
					innerCos: -2,
					color: [1, 1, 1],
					castsShadow: false,
					shadowIndex: 0,
				},
				{
					type: 0,
					position: [2, 0, -7],
					range: 5,
					direction: [0, 0, 0],
					outerCos: -2,
					innerCos: -2,
					color: [1, 0, 0],
					castsShadow: false,
					shadowIndex: 0,
				},
			],
		});

		runtime.prepare(context, lights, 4096);

		assert.ok(
			warnings.some((warning) =>
				warning.includes("clamps lights to 1; extra lights are skipped")
			)
		);
	});
}

function testClusterHeadersUseFixedClusterSpans() {
	const gl = createFakeGL();
	const runtime = new WebGLClusteredLightingRuntime(gl);
	const context = createPerspectiveContext({
		attachments: {
			width: 128,
			height: 64,
		},
		camera: {
			type: CameraType.Perspective,
			near: 0.1,
			far: 200,
			fov: 60,
			aspectRatio: 2,
			viewMatrix: Matrix4.identity(),
			viewProjectionMatrix: Matrix4.perspective(60, 2, 0.1, 200),
		},
		features: {
			clusteredLightingOptions: {
				tileSizePx: 64,
				zSlices: 1,
				maxLights: 4,
				maxLightsPerCluster: 2,
			},
		},
	});

	runtime.prepare(context, createLightState(), 4096);
	const state = runtime.getState();
	const headerUpload = gl.getUpload(state.headerTexture);

	assert.equal(state.enabled, true);
	assert.ok(headerUpload);
	assert.equal(headerUpload.pixels[0], 0);
	assert.equal(headerUpload.pixels[4], 2);
}

function testClusterOverflowClampsFixedSpanCount() {
	const gl = createFakeGL();
	const runtime = new WebGLClusteredLightingRuntime(gl);
	const context = createPerspectiveContext({
		attachments: {
			width: 64,
			height: 64,
		},
		features: {
			clusteredLightingOptions: {
				tileSizePx: 64,
				zSlices: 1,
				maxLights: 4,
				maxLightsPerCluster: 1,
			},
		},
	});
	const lights = createLightState({
		clusteredLights: [
			{
				type: 0,
				position: [0, 0, -6],
				range: 5,
				direction: [0, 0, 0],
				outerCos: -2,
				innerCos: -2,
				color: [1, 1, 1],
				castsShadow: false,
				shadowIndex: 0,
			},
			{
				type: 0,
				position: [0, 0, -6],
				range: 5,
				direction: [0, 0, 0],
				outerCos: -2,
				innerCos: -2,
				color: [1, 0, 0],
				castsShadow: false,
				shadowIndex: 0,
			},
		],
	});

	runtime.prepare(context, lights, 4096);
	const state = runtime.getState();
	const headerUpload = gl.getUpload(state.headerTexture);
	const indexUpload = gl.getUpload(state.indexTexture);

	assert.equal(state.enabled, true);
	assert.ok(headerUpload);
	assert.ok(indexUpload);
	assert.equal(headerUpload.pixels[0], 0);
	assert.equal(headerUpload.pixels[1], 1);
	assert.equal(headerUpload.pixels[2], 1);
	assert.equal(indexUpload.pixels[0], 0);
}

function run() {
	testPerspectiveBuildsClusterTextures();
	testNonPerspectiveFallsBackWithWarning();
	testLightBudgetWarning();
	testClusterHeadersUseFixedClusterSpans();
	testClusterOverflowClampsFixedSpanCount();
	console.log("WebGL clustered lighting runtime tests passed");
}

run();
