import assert from "node:assert/strict";
import { WebGPUFrameBindingCache } from "../../../src/backends/webgpu/WebGPUFrameBindingCache.ts";
import {
	WebGPUFrameServiceOwner as WebGPURenderResources
} from "../../../src/backends/webgpu/WebGPUFrameServiceOwner.ts";
import {
	ShaderSource
} from "../../../src/shaders/ShaderSource.ts";
import {
	packMatrix4ForWGSL
} from "../../../src/backends/webgpu/index.ts";
import {
	resolveFeatureState
} from "../../../src/pipeline/FeatureResolver.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";
import {
	ReflectionProbe
} from "../../../src/lights/ReflectionProbe.ts";
import {
	Matrix4
} from "../../../src/maths/Matrix4.ts";
import {
	computeHaltonJitterNDC
} from "../../../src/maths/Misc.ts";
import {
	SH
} from "../../../src/maths/SH.ts";
import {
	PBRMaterial
} from "../../../src/materials/PBRMaterial.ts";
import {
	Logger
} from "../../../src/foundation/Logger.ts";
import {
	createWebGPUComputeFacade
} from "../../../src/backends/webgpu/ComputeFacade.ts";
import {
	createResolvedPostProcess
} from "../../helpers/postprocess.mjs";


import {
	FakeWebGPUBackend as FakeBackend,
} from "../../helpers/fakes.mjs";
import {
	assertArrayNearlyEqual,
	createEnvironmentSnapshot,
	createFrame,
	createFrameContext,
	createFrameScopeAdapter,
	createMainFrameOptions,
	createModel,
	createPacket,
	createPreparedFrameResources,
	createTinyTexture,
	createWebGPUFrameContextForTemporalTest,
	readLatestFrameCameraUniformField
} from "../../helpers/webgpu-bridge.mjs";
const previousGPUShaderStage = globalThis.GPUShaderStage;
globalThis.GPUShaderStage = {
	...(previousGPUShaderStage ?? {}),
	VERTEX: previousGPUShaderStage?.VERTEX ?? 1,
	FRAGMENT: previousGPUShaderStage?.FRAGMENT ?? 2,
	COMPUTE: previousGPUShaderStage?.COMPUTE ?? 4,
};
ShaderSource.resetConfiguration();
Logger.reset();

async function testWebGPUEnvironmentCombinationsRegression() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const baseScene = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const caps = {
		sh: true,
		shadows: true,
		reflection: false,
		environment: true,
		ssao: false,
		taa: false,
		ssr: false,
		volumetric: false,
		fog: false,
		motionBlur: false,
		dof: false,
		bloom: false,
		clusteredLighting: true,
	};

	const shAmbient = SH.empty();
	shAmbient[0] = { r: 12, g: 12, b: 12 };
	const probeMap = createTinyTexture(2);
	const probe = new ReflectionProbe({
		shape: "box",
		prefilteredMap: probeMap,
	});

	const cases = [
		{
			environment: createEnvironmentSnapshot(
				createTinyTexture(1),
				createTinyTexture(1)
			),
			lights: [probe],
			enableSH: true,
			expectEnvironment: true,
		},
		{
			environment: createEnvironmentSnapshot(null, null),
			lights: [probe],
			enableSH: true,
			expectEnvironment: false,
		},
		{
			environment: createEnvironmentSnapshot(null, null),
			lights: [],
			enableSH: false,
			expectEnvironment: false,
		},
	];

	for (const scenario of cases) {
		const scene = {
			...baseScene,
			environment: scenario.environment,
			lights: scenario.lights,
		};
		const features = resolveFeatureState(
			{
				enableLighting: true,
				enableGamma: true,
				enableSH: scenario.enableSH,
				enableShadows: true,
				enableEnvironment: true,
			},
			caps,
			"webgpu"
		);
		const frameResources = resources.prepareFrame(
			{
				viewCamera: scene.camera,
				attachments: { width: 16, height: 16 },
				features,
				postProcess: createResolvedPostProcess(),
				shadowMaps: scene.shadowMaps,
				scene,
				shCoeffs: SH.empty(),
				shAmbientCoeffs: scenario.enableSH ? shAmbient : SH.empty(),
				worldMatrix: Matrix4.identity(),
				transient: new Map(),
			},
			createMainFrameOptions()
		);

		const environmentResources = await resources.getEnvironmentResources(
			frameResources,
			"mrt",
			{ sampleCount: 1 },
		);
		assert.equal(!!environmentResources, scenario.expectEnvironment);
		const draw = await resources.getDrawResources(packet, frameResources, {
			sampleCount: 1,
		});
		assert.ok(draw);
		resources.commitTemporalFrame();
	}
}

async function testScopedSceneTargetModesUseDistinctBindings() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	backend.canvasDepthFormat = "depth24plus";
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.environment = createEnvironmentSnapshot(
		createTinyTexture(1),
		createTinyTexture(1)
	);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: true,
			enableEnvironment: true,
			enableClusteredLighting: true,
		},
		{
			sh: false,
			shadows: true,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);

	const mainFrameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "main",
			sceneTargetMode: "single",
		}
	);
	const mainFrameBinding = mainFrameResources.frameBinding;
	const captureFrameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "test-capture",
			sceneTargetMode: "mrt",
			temporalStateMode: "disabled",
		}
	);
	assert.notEqual(captureFrameResources.frameBinding, mainFrameBinding);

	const captureDrawResources = await resources.getDrawResources(
		packet,
		captureFrameResources,
		{ sceneTargetMode: "mrt", sampleCount: 1 },
	);
	assert.ok(captureDrawResources);
	assert.equal(captureDrawResources[0].pipeline.label.endsWith("_mrt"), true);
	assert.equal(mainFrameResources.frameBinding, mainFrameBinding);

	const environmentResources = await resources.getEnvironmentResources(
		mainFrameResources,
		"single",
		{ sampleCount: 1 },
	);
	assert.ok(environmentResources);
	assert.equal(
		environmentResources.pipeline.label,
		"WebGPUEnvironmentPipeline_single"
	);
	assert.equal(
		environmentResources.pipeline.desc.depthStencil.format,
		backend.canvasDepthFormat
	);

	const drawResources = await resources.getDrawResources(
		packet,
		mainFrameResources,
		{ sceneTargetMode: "single", sampleCount: 1 },
	);
	assert.ok(drawResources);
	assert.equal(drawResources[0].pipeline.label.endsWith("_single"), true);
	assert.equal(
		drawResources[0].pipeline.desc.depthStencil.format,
		backend.canvasDepthFormat
	);
}

async function testSampleCountOverrideUsesSingleSampleCapturePipelines() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	backend.canvasDepthFormat = "depth24plus";
	const msaa = {
		sampleCount: 4,
		resolveSupportedSampleCount: (requested) => Math.max(1, Math.floor(requested)),
		fallbackToSingleSample: () => false,
	};
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	frame.environment = createEnvironmentSnapshot(
		createTinyTexture(1),
		createTinyTexture(1)
	);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend), msaa);

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableShadows: false,
			enableEnvironment: true,
			enableClusteredLighting: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const frameResources = resources.prepareFrame(
		createFrameContext(frame, features),
		{
			scopeKey: "main",
			sceneTargetMode: "mrt",
		}
	);

	const mainDrawResources = await resources.getDrawResources(
		packet,
		frameResources,
		{ sceneTargetMode: "mrt", sampleCount: 4 },
	);
	assert.ok(mainDrawResources);
	assert.equal(mainDrawResources[0].pipeline.desc.sampleCount, 4);

	const captureDrawResources = await resources.getDrawResources(
		packet,
		frameResources,
		{
			sceneTargetMode: "mrt",
			drawMode: "reflection-capture",
			sampleCount: 1,
		}
	);
	assert.ok(captureDrawResources);
	assert.equal(captureDrawResources[0].pipeline.desc.sampleCount, 1);

	const mainEnvironment = await resources.getEnvironmentResources(
		frameResources,
		"mrt",
		{ sampleCount: 4 },
	);
	assert.ok(mainEnvironment);
	assert.equal(mainEnvironment.pipeline.desc.sampleCount, 4);

	const captureEnvironment =
		await resources.getEnvironmentResources(frameResources, "mrt", {
			sampleCount: 1,
		});
	assert.ok(captureEnvironment);
	assert.equal(captureEnvironment.pipeline.desc.sampleCount, 1);
}

async function testFrameBindingReplacementDestroysOldBinding() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const frame = createFrame(packet);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableEnvironment: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: true,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);

	const firstFrameResources = resources.prepareFrame(
		createFrameContext(
			{
				...frame,
				environment: createEnvironmentSnapshot(
					createTinyTexture(1),
					createTinyTexture(1)
				),
			},
			features
		),
		createMainFrameOptions()
	);
	const firstEnvironment =
		await resources.getEnvironmentResources(
			firstFrameResources,
			"mrt",
			{ sampleCount: 1 },
		);
	assert.ok(firstEnvironment);
	const firstBinding = firstEnvironment.frameBinding;
	assert.equal(firstBinding.destroyed, false);
	resources.commitTemporalFrame();

	const secondFrameResources = resources.prepareFrame(
		createFrameContext(
			{
				...frame,
				environment: createEnvironmentSnapshot(
					createTinyTexture(1),
					createTinyTexture(1)
				),
			},
			features
		),
		createMainFrameOptions()
	);
	assert.equal(firstBinding.destroyed, true);

	const secondEnvironment =
		await resources.getEnvironmentResources(
			secondFrameResources,
			"mrt",
			{ sampleCount: 1 },
		);
	assert.ok(secondEnvironment);
	assert.notEqual(secondEnvironment.frameBinding, firstBinding);
}

async function testWebGPUPrepareFrameTemporalStateModes() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: true,
			ssr: false,
			volumetric: false,
			fog: false,
			motionBlur: false,
			dof: false,
			bloom: false,
			clusteredLighting: true,
		},
		"webgpu"
	);
	const postProcess = createResolvedPostProcess(
		{ taa: { enabled: true, options: { jitterScale: 1 } } },
		"webgpu"
	);
	const width = 16;
	const height = 8;
	const temporalFrameRequirements = {
		cameraJitter: { sequence: "halton-2-3", scale: 1 },
	};
	const previousViewProjection = new Matrix4([
		[1, 0, 0, 10],
		[0, 1, 0, 20],
		[0, 0, 1, 30],
		[0, 0, 0, 1],
	]);
	const currentViewProjection = new Matrix4([
		[2, 0, 0, 40],
		[0, 2, 0, 50],
		[0, 0, 2, 60],
		[0, 0, 0, 1],
	]);
	const captureViewProjection = new Matrix4([
		[3, 0, 0, 70],
		[0, 3, 0, 80],
		[0, 0, 3, 90],
		[0, 0, 0, 1],
	]);

	const previousFrame = createFrame(packet);
	previousFrame.camera.viewProjectionMatrix = previousViewProjection;
	const currentFrame = createFrame(packet);
	currentFrame.camera.viewProjectionMatrix = currentViewProjection;
	const captureFrame = createFrame(packet);
	captureFrame.camera.viewProjectionMatrix = captureViewProjection;

	const previousContext = createWebGPUFrameContextForTemporalTest(
		previousFrame,
		features,
		postProcess,
		width,
		height
	);
	resources.prepareFrame(previousContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const firstJitter = computeHaltonJitterNDC(0, width, height, 1);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[firstJitter[0], firstJitter[1], 0, 0]
	);
	resources.commitTemporalFrame();

	const currentContext = createWebGPUFrameContextForTemporalTest(
		currentFrame,
		features,
		postProcess,
		width,
		height
	);
	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const secondJitter = computeHaltonJitterNDC(1, width, height, 1);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(previousViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[secondJitter[0], secondJitter[1], firstJitter[0], firstJitter[1]]
	);

	const captureContext = createWebGPUFrameContextForTemporalTest(
		captureFrame,
		features,
		postProcess,
		width,
		height,
		true
	);
	resources.prepareFrame(
		captureContext,
		createMainFrameOptions({
			scopeKey: "capture",
			temporalStateMode: "disabled",
			frameRequirements: temporalFrameRequirements,
		})
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(captureViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[0, 0, 0, 0]
	);

	resources.prepareFrame(
		currentContext,
		createMainFrameOptions({
			temporalStateMode: "reuse",
			frameRequirements: temporalFrameRequirements,
		})
	);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(previousViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[secondJitter[0], secondJitter[1], firstJitter[0], firstJitter[1]]
	);
	resources.commitTemporalFrame();

	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const thirdJitter = computeHaltonJitterNDC(2, width, height, 1);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[thirdJitter[0], thirdJitter[1], secondJitter[0], secondJitter[1]]
	);
	resources.commitTemporalFrame();

	const resetContext = createWebGPUFrameContextForTemporalTest(
		currentFrame,
		features,
		postProcess,
		width,
		height,
		true
	);
	resources.prepareFrame(resetContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(currentViewProjection))
	);
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		[firstJitter[0], firstJitter[1], 0, 0]
	);

	resources.commitTemporalFrame();
	resources.beginFrameResourceLifecycle();
	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const abortedJitter = readLatestJitter(backend);
	resources.abortTemporalFrame();
	resources.beginFrameResourceLifecycle();
	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	assertArrayNearlyEqual(
		readLatestJitter(backend),
		abortedJitter,
	);
	resources.abortTemporalFrame();
}

async function testSceneFrameBindingLayoutMatchesShaderContract() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const sceneLayout = backend.device.bindGroupLayouts.find(
		(layout) => layout.desc.label === "WebGPUSceneFrameBindGroupLayout"
	);
	assert.ok(sceneLayout);
	assert.equal(sceneLayout.desc.entries.length, 12);
	assert.deepEqual(
		sceneLayout.desc.entries.map((entry) => entry.binding),
		[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]
	);
	assert.equal(sceneLayout.desc.entries[4].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[5].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[6].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[7].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[8].sampler?.type, "comparison");
	assert.equal(sceneLayout.desc.entries[9].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[10].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[11].buffer?.type, "uniform");

	resources.destroy();
}

function readLatestJitter(backend) {
	return [
		...readLatestFrameCameraUniformField(backend, "jitterNdc"),
		...readLatestFrameCameraUniformField(backend, "prevJitterNdc"),
	];
}

function createFrameBindingFixture() {
	const backend = new FakeBackend();
	const token = {};
	const registry = {
		getWhiteTexture: () => token, getWhiteSampler: () => token,
		getTextureForSlot: () => token, getSamplerForTexture: () => token,
	};
	const shadow = { atlas: null, transmittanceAtlas: null };
	const cache = new WebGPUFrameBindingCache(backend, null, null, registry, shadow);
	const frame = createFrame(createPacket(createModel([new PBRMaterial()])));
	const lighting = { ambientColor: [0.1, 0.2, 0.3], directionalLights: [],
		pointLights: [], spotLights: [], areaLights: [], directionalShadows: [], spotShadows: [] };
	const features = { enableLighting: true, enableShadows: false,
		enableClusteredLighting: false, postProcess: createResolvedPostProcess() };
	const environment = { shAmbientCoeffs: null, localLightProbeCount: 0,
		localLightProbes: [], irradianceProbeGrid: null, reflectionProbeCount: 0,
		reflectionProbes: [], brdfLUTTexture: null, envSpecularMaxMipLevel: 0 };
	return { backend, cache, frame, lighting, features, environment };
}

function testCameraUniformInversesAndFailureRecovery() {
	const { backend, cache, frame, lighting, features, environment } = createFrameBindingFixture();
	const secondary = createFrameBindingFixture();
	const replacementCamera = secondary.frame.camera;
	secondary.frame.camera = frame.camera;
	const prepare = () => cache.prepare(frame, lighting, environment, features, 16, 8, "mrt",
		{ temporalStateMode: "disabled" });
	const inverse = Matrix4.inverse;
	let calls = 0;
	Matrix4.inverse = (...args) => { calls++; return inverse(...args); };
	try {
		prepare();
		assert.equal(calls, 2, "frame preparation only inverts the view and projection matrices");
		secondary.cache.prepare(secondary.frame, lighting, environment, features, 16, 8, "mrt",
			{ temporalStateMode: "disabled" });
		assert.equal(calls, 4, "each view scope reads freshly computed camera inverses");
		frame.camera.viewMatrix = frame.camera.viewMatrix.clone();
		frame.camera.projectionMatrix = frame.camera.projectionMatrix.clone();
		frame.camera.viewProjectionMatrix = frame.camera.viewProjectionMatrix.clone();
		prepare();
		assert.equal(calls, 6, "unchanged numeric values are inverted on each prepare");
		frame.camera.viewMatrix.elements[0][3] = 7;
		prepare();
		assert.equal(calls, 8);
		assert.equal(readLatestFrameCameraUniformField(backend, "inverseView")[12], -7);
		frame.camera.projectionMatrix.elements[0][0] = 2;
		prepare();
		assert.equal(calls, 10);
		assert.equal(readLatestFrameCameraUniformField(backend, "inverseProjection")[0], 0.5);
		const previousView = frame.camera.viewMatrix;
		frame.camera.viewMatrix = new Matrix4([
			[1, 1e38, 0, 0], [0, 1, 1e38, 0], [0, 0, 1, 1e38], [0, 0, 0, 1],
		]);
		const writesBeforeFailure = backend.writeBufferCalls;
		assert.throws(prepare, /view inverse.*finite/i);
		assert.equal(backend.writeBufferCalls, writesBeforeFailure);
		frame.camera.viewMatrix = previousView;
		prepare();
		assertArrayNearlyEqual(readLatestFrameCameraUniformField(backend, "inverseView"),
			Array.from(packMatrix4ForWGSL(inverse(previousView))));
		replacementCamera.viewMatrix.elements[0][3] = 11;
		frame.camera = replacementCamera;
		prepare();
		assert.equal(readLatestFrameCameraUniformField(backend, "inverseView")[12], -11,
			"a view scope switching cameras reads the new camera's inverses");
	} finally {
		Matrix4.inverse = inverse;
		cache.destroy();
		secondary.cache.destroy();
	}
}

function testIndependentEnvironmentFlagsAndWriterReuse() {
	const { backend, cache, frame, lighting, features, environment } = createFrameBindingFixture();
	const latestData = (label) => backend.writeCalls.findLast((call) => call[1]?.label === label)[2];
	let firstCameraData;
	for (const [backgroundSpace, iblSpace, probeCount, enableSH, hasSH] of [
		["sRGB", "linear", 0, true, false],
		["linear", "sRGB", 0, true, true],
		["sRGB", "sRGB", 1, false, true],
		[null, null, 0, false, false],
	]) {
		environment.environmentTexture = backgroundSpace ? { colorSpace: backgroundSpace } : null;
		environment.envSpecularTexture = iblSpace ? { colorSpace: iblSpace } : null;
		environment.reflectionProbeCount = probeCount;
		environment.enableSH = enableSH;
		environment.hasSHAmbient = hasSH;
		environment.envSpecularMaxMipLevel = 3.5;
		cache.prepare(frame, lighting, environment, features, 16, 8, "mrt", { temporalStateMode: "disabled" });
		const cameraData = latestData("WebGPUFrameCameraUniforms");
		if (firstCameraData) assert.equal(cameraData, firstCameraData, "scope reuses camera buffer storage");
		firstCameraData = cameraData;
		const environmentData = latestData("WebGPUFrameEnvironmentUniforms");
		const flags = new Uint32Array(environmentData);
		assert.equal(flags[0], enableSH && hasSH ? 1 : 0);
		assert.equal(flags[1], iblSpace ? 1 : 0);
		assert.equal(environmentData.byteLength, 4224);
		assert.equal(new Float32Array(environmentData)[3], 3.5);
		const background = latestData("WebGPUEnvironmentBackgroundParams");
		assert.equal(background.byteLength, 16);
		assert.equal(backend.bufferDescs.findLast((desc) =>
			desc.label === "WebGPUEnvironmentBackgroundParams").size, 16);
	}
	// An invalid next camera must not change a committed history or start a transaction.
	cache.prepare(frame, lighting, environment, features, 16, 8, "mrt");
	cache.commitTemporalFrame();
	const writes = backend.writeBufferCalls;
	frame.camera.viewMatrix.elements[3][3] = 0;
	assert.throws(() => cache.prepare(frame, lighting, environment, features, 16, 8, "mrt"), /view.*singular/i);
	assert.equal(backend.writeBufferCalls, writes);
	assert.equal(cache._currentTemporalSnapshot, null);
	frame.camera.viewMatrix.elements[3][3] = 1;
	frame.camera.viewProjectionMatrix.elements[0][3] = 9;
	cache.prepare(frame, lighting, environment, features, 16, 8, "mrt");
	assertArrayNearlyEqual(readLatestFrameCameraUniformField(backend, "prevViewProjection"),
		Array.from(packMatrix4ForWGSL(Matrix4.identity())));
	cache.abortTemporalFrame();
	cache.destroy();
}

function testInvalidCameraBeforeTemporalOrGPUWrites() {
	const backend = new FakeBackend();
	const cache = new WebGPUFrameBindingCache(backend, null, null, null, null);
	const frame = createFrame(createPacket(createModel([new PBRMaterial()])));
	frame.camera.projectionMatrix = Matrix4.identity();
	frame.camera.projectionMatrix.elements[0][0] = 0;
	assert.throws(() => cache.prepare(frame, null, null, null, 16, 8, "mrt"),
		/Camera projection matrix.*singular/i);
	assert.equal(backend.writeBufferCalls, 0);
	assert.equal(cache._currentTemporalSnapshot, null);
	frame.camera.projectionMatrix.elements[0][0] = Infinity;
	assert.throws(() => cache.prepare(frame, null, null, null, 16, 8, "mrt"),
		/WebGPU camera projection matrix.*finite/i);
	assert.equal(backend.writeBufferCalls, 0);
	assert.equal(cache._currentTemporalSnapshot, null);
	frame.camera.projectionMatrix = Matrix4.identity();
	frame.camera.viewProjectionMatrix.elements[0][0] = Infinity;
	assert.throws(() => cache.prepare(frame, null, null, null, 16, 8, "mrt"),
		/WebGPU camera view-projection matrix.*finite/i);
	assert.equal(backend.writeBufferCalls, 0);
	assert.equal(cache._currentTemporalSnapshot, null);
}

async function run() {
	try {
		testInvalidCameraBeforeTemporalOrGPUWrites();
		testCameraUniformInversesAndFailureRecovery();
		testIndependentEnvironmentFlagsAndWriterReuse();
		await testWebGPUEnvironmentCombinationsRegression();
		await testScopedSceneTargetModesUseDistinctBindings();
		await testSampleCountOverrideUsesSingleSampleCapturePipelines();
		await testFrameBindingReplacementDestroysOldBinding();
		await testWebGPUPrepareFrameTemporalStateModes();
		await testSceneFrameBindingLayoutMatchesShaderContract();
		console.log("WebGPU bridge capture/temporal tests passed");
	} finally {
		ShaderSource.resetConfiguration();
		Logger.reset();
		if (previousGPUShaderStage === undefined) {
			delete globalThis.GPUShaderStage;
		} else {
			globalThis.GPUShaderStage = previousGPUShaderStage;
		}
	}
}
await run();
