import assert from "node:assert/strict";
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
	WebGPUReflectionProbeCapturePass
} from "../../../src/backends/webgpu/WebGPUReflectionProbeCapturePass.ts";
import {
	resolveFeatureState
} from "../../../src/pipeline/FeatureResolver.ts";
import {
	PreparedSceneBuilder
} from "../../../src/pipeline/PreparedSceneBuilder.ts";
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
	Node
} from "../../../src/core/Node.ts";
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

async function testReflectionProbeCaptureUsesLegacyMRTAttachmentFormats() {
	const backend = new FakeBackend();
	backend.canvasFormat = "bgra8unorm";
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const preparedScene = {
		...createFrame(packet),
		particleSystems: [],
		hasActiveAnimations: false,
		spatialIndex: null,
	};
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableClusteredLighting: true,
			enableEnvironment: false,
			enableShadows: false,
			enableReflection: false,
			enableOIT: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			oit: false,
			ssao: false,
			ssgi: false,
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
	const frameContext = {
		viewCamera: preparedScene.camera,
		attachments: { width: 1, height: 1 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: preparedScene.shadowMaps,
		scene: preparedScene,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 1, height: 1 }],
			dirtyTileSize: 1,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient: new Map(),
	};
	let preparedCaptureContext = null;
	const resources = {
		createFrameScope() { return createFrameScopeAdapter(resources); },
		prepareFrame(context, options = {}) {
			preparedCaptureContext = context;
			return createPreparedFrameResources(options);
		},
		releaseScope() {},
		async buildClusteredLighting() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {
			return 0;
		},
		getParticleBillboardRenderer() {
			return this;
		},
	};
	const probe = new ReflectionProbe({
		includeMeshes: true,
		includeEnvironment: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	const readyComputeFacade = createWebGPUComputeFacade(backend);
	backend.computeFacade = new Proxy(readyComputeFacade, {
		get(target, property, receiver) {
			if (property === "device" || property === "queue") {
				return null;
			}
			return Reflect.get(target, property, receiver);
		},
	});
	const capturePass = new WebGPUReflectionProbeCapturePass(
		backend,
		resources,
	);
	backend.computeFacade = readyComputeFacade;
	const probeCache = probe.getRuntimeCache();
	const originalRebuildForCamera = PreparedSceneBuilder.rebuildForCamera;
	const transparentPacket = {
		...packet,
		id: `${packet.id}:transparent`,
	};
	let rebuildInput = null;
	PreparedSceneBuilder.rebuildForCamera = (source, camera, options) => {
		rebuildInput = { source, camera, options };
		return {
			...source,
			camera,
			opaquePackets: [packet],
			transparentPackets: [transparentPacket],
			reflectivePackets: [packet],
			decalPackets: [{ id: "capture-decal" }],
		};
	};
	let result;
	try {
		result = await capturePass.captureFace({
			frameContext,
			targetId: probe.id,
			targetKind: "reflection",
			captureWorldPosition: probeCache.captureWorldPosition,
			captureFar: probe.captureFar,
			faceIndex: 0,
			faceSize: 1,
			includeEnvironment: false,
			includeMeshes: true,
			includeTransparent: false,
			includeParticles: false,
			includeShadows: false,
		});
	} finally {
		PreparedSceneBuilder.rebuildForCamera = originalRebuildForCamera;
	}

	assert.ok(result);
	assert.equal(result.length, 4);
	assert.strictEqual(rebuildInput.source, preparedScene);
	assert.strictEqual(rebuildInput.camera, preparedCaptureContext.viewCamera);
	assert.equal(rebuildInput.options.visibilityScene, null);
	assert.strictEqual(
		preparedCaptureContext.scene.camera,
		preparedCaptureContext.viewCamera
	);
	assert.deepEqual(preparedCaptureContext.scene.opaquePackets, [packet]);
	assert.deepEqual(preparedCaptureContext.scene.transparentPackets, []);
	assert.strictEqual(
		preparedCaptureContext.scene.shadowCasterPackets,
		preparedScene.shadowCasterPackets
	);
	assert.strictEqual(
		preparedCaptureContext.scene.shadowTransmitterPackets,
		preparedScene.shadowTransmitterPackets
	);
	assert.deepEqual(preparedCaptureContext.scene.reflectivePackets, []);
	assert.deepEqual(preparedCaptureContext.scene.decalPackets, []);
	assert.equal(backend.createTextureCalls.length >= 2, true);
	assert.deepEqual(
		[
			"SceneColor",
			"Albedo",
			"Normal",
			"Emissive",
			"Motion",
		].map((target) =>
			backend.createTextureCalls.find((call) =>
				call.label === `WebGPUReflectionProbeCapture${target}_face0`
			)?.format
		),
		[
			TextureFormat.RGBA16Float,
			TextureFormat.RGBA8Unorm,
			TextureFormat.RGBA8Unorm,
			TextureFormat.RGBA16Float,
			TextureFormat.RGBA16Float,
		]
	);
	assert.equal(
		backend.createTextureCalls.some(
			(call) => call.format === TextureFormat.Depth32Float
		),
		true
	);
}

async function testReflectionProbeCaptureUsesParentWorldPositionAsOrigin() {
	const backend = new FakeBackend();
	const model = createModel([new PBRMaterial()]);
	const packet = createPacket(model);
	const preparedScene = {
		...createFrame(packet),
		particleSystems: [],
		hasActiveAnimations: false,
		spatialIndex: null,
	};
	const features = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableClusteredLighting: true,
			enableEnvironment: false,
			enableShadows: false,
			enableReflection: false,
			enableOIT: false,
			enableSSAO: false,
			enableSSGI: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFog: false,
			enableMotionBlur: false,
			enableDOF: false,
			enableBloom: false,
		},
		{
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			oit: false,
			ssao: false,
			ssgi: false,
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
	const frameContext = {
		camera: preparedScene.camera,
		attachments: { width: 1, height: 1 },
		features,
		postProcess: createResolvedPostProcess(),
		shadowMaps: preparedScene.shadowMaps,
		scene: preparedScene,
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [{ x: 0, y: 0, width: 1, height: 1 }],
			dirtyTileSize: 1,
			dirtyTileColumns: 1,
			dirtyTileRows: 1,
			dirtyTiles: [0],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient: new Map(),
	};
	const preparedCameraPositions = [];
	const resources = {
		createFrameScope() { return createFrameScopeAdapter(resources); },
		prepareFrame(context, options = {}) {
			preparedCameraPositions.push(
				context.viewCamera.getWorldPosition({ x: 0, y: 0, z: 0 })
			);
			return createPreparedFrameResources(options);
		},
		releaseScope() {},
		async buildClusteredLighting() {},
		async getEnvironmentResources() {
			return null;
		},
		async getDrawResources() {
			return null;
		},
		async renderParticles() {
			return 0;
		},
		getParticleBillboardRenderer() {
			return this;
		},
	};
	const modelRoot = new Node();
	modelRoot.position.set(3, 0, 0);
	const probe = new ReflectionProbe({
		includeMeshes: false,
		includeEnvironment: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	modelRoot.addChild(probe);
	probe.position.set(2, 0, 0);
	modelRoot.updateWorldMatrix();
	backend.computeFacade = createWebGPUComputeFacade(backend);
	const capturePass = new WebGPUReflectionProbeCapturePass(
		backend,
		resources,
	);
	const probeCache = probe.getRuntimeCache();

	await capturePass.captureFace({
		frameContext,
		targetId: probe.id,
		targetKind: "reflection",
		captureWorldPosition: probeCache.captureWorldPosition,
		captureFar: probe.captureFar,
		faceIndex: 0,
		faceSize: 1,
		includeEnvironment: false,
		includeMeshes: false,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});

	assert.ok(preparedCameraPositions.length >= 1);
	assert.deepEqual(preparedCameraPositions[0], { x: 3, y: 0, z: 0 });
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
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
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
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
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
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
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
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[secondJitter[0], secondJitter[1], firstJitter[0], firstJitter[1]]
	);
	resources.commitTemporalFrame();

	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const thirdJitter = computeHaltonJitterNDC(2, width, height, 1);
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
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
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		[firstJitter[0], firstJitter[1], 0, 0]
	);

	resources.commitTemporalFrame();
	resources.beginFrameResourceLifecycle();
	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	const abortedJitter = readLatestFrameCameraUniformField(
		backend,
		"taaJitterCurrentPrev",
	);
	resources.abortTemporalFrame();
	resources.beginFrameResourceLifecycle();
	resources.prepareFrame(currentContext, createMainFrameOptions({
		frameRequirements: temporalFrameRequirements,
	}));
	assertArrayNearlyEqual(
		readLatestFrameCameraUniformField(backend, "taaJitterCurrentPrev"),
		abortedJitter,
	);
	resources.abortTemporalFrame();
}

async function testSceneFrameBindingLayoutMatchesFallbackEnvironmentContract() {
	const backend = new FakeBackend();
	const resources = new WebGPURenderResources(backend, backend, createWebGPUComputeFacade(backend));

	const sceneLayout = backend.device.bindGroupLayouts.find(
		(layout) => layout.desc.label === "WebGPUSceneFrameBindGroupLayout"
	);
	assert.ok(sceneLayout);
	assert.equal(sceneLayout.desc.entries.length, 17);
	assert.deepEqual(
		sceneLayout.desc.entries.map((entry) => entry.binding),
		[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
	);
	assert.equal(sceneLayout.desc.entries[4].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[5].sampler?.type, "filtering");
	assert.equal(sceneLayout.desc.entries[6].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[7].buffer?.type, "read-only-storage");
	assert.equal(sceneLayout.desc.entries[8].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[9].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[10].texture?.sampleType, "float");
	assert.equal(sceneLayout.desc.entries[11].texture?.sampleType, "uint");
	assert.equal(sceneLayout.desc.entries[12].texture?.sampleType, "depth");
	assert.equal(sceneLayout.desc.entries[13].sampler?.type, "comparison");
	assert.equal(sceneLayout.desc.entries[14].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[15].buffer?.type, "uniform");
	assert.equal(sceneLayout.desc.entries[16].buffer?.type, "uniform");

	resources.destroy();
}

async function run() {
	try {
		await testWebGPUEnvironmentCombinationsRegression();
		await testScopedSceneTargetModesUseDistinctBindings();
		await testSampleCountOverrideUsesSingleSampleCapturePipelines();
		await testReflectionProbeCaptureUsesLegacyMRTAttachmentFormats();
		await testReflectionProbeCaptureUsesParentWorldPositionAsOrigin();
		await testFrameBindingReplacementDestroysOldBinding();
		await testWebGPUPrepareFrameTemporalStateModes();
		await testSceneFrameBindingLayoutMatchesFallbackEnvironmentContract();
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
