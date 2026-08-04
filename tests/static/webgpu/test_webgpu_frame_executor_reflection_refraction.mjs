import assert from "node:assert/strict";

import * as frameExecutorFixture from "../../helpers/webgpu_frame_executor_resilience.mjs";

const {
	Camera,
	FakeBackend,
	Material,
	WebGPUFrameExecutor,
	createFrameContext,
	createMSAAContext,
	createOITBackend,
	createOITSequencingResourcesStub,
	createPlanarPacket,
	createPlanarReflectionResourcesStub,
	createResolvedPostProcess,
	getFrameGraphDebugState,
	getFrameTargets,
	getMSAATargets,
	initializeIsolatedWebGPUTestState,
} = frameExecutorFixture;

const restoreTestState = initializeIsolatedWebGPUTestState();

async function testPlanarReflectionCaptureAndCompositeSequencing() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	const colorTarget = backend.createTextureCalls.find((desc) =>
		String(desc.label).startsWith("WebGPUPlanarReflectionColor_")
	);
	assert.equal(colorTarget.width, 32);
	assert.equal(colorTarget.height, 32);
	assert.ok(
		resources._state.events.includes(
			"prepare:reflection:false:ssr:false:opaque:object"
		)
	);
	assert.ok(
		resources._state.events.includes("draw:object:color:reflection-capture")
	);
	assert.ok(
		resources._state.environmentOptions.some(
			(entry) => entry.sceneTargetMode === "color"
		)
	);
	assert.ok(
		resources._state.events.includes(
			"draw:mirror:mrt:planar-reflection-composite"
		)
	);
	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	const capturePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionCaptureMain"
	);
	assert.ok(capturePass);
	assert.equal(capturePass.colorAttachments.length, 1);
	assert.ok(
		labels.indexOf("WebGPUPlanarReflectionCaptureMain") <
			labels.indexOf("WebGPUMainMRT_Clear")
	);
	assert.ok(
		labels.indexOf("WebGPUPlanarReflectionComposite") >
			labels.indexOf("WebGPUMainMRT_Clear")
	);
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.equal(compositePass.colorAttachments.length, 2);
	assert.equal(
		String(compositePass.colorAttachments[1].view.label).startsWith(
			"WebGPUPlanarReflectionMask"
		),
		true
	);
	assert.equal(backend.submits, 0);
	await executor.endFrame();
	assert.deepEqual(getFrameGraphDebugState(executor).commit.submittedLabels, [
		"main:before-reflection",
		"planar-reflection:0.000000,1.000000,0.000000,0.000000",
		"main:final",
	]);
}

async function testPlanarReflectionUsesColorTargetsWithoutPostProcess() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({});
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	assert.equal(executor.getSceneTargetModeForFrame(), "color");
	assert.equal(targets.postPing, null);
	assert.equal(targets.postPong, null);
	assert.equal(targets.gAlbedoAlpha, null);
	assert.equal(targets.gMotionDepth, null);
	assert.ok(targets.planarReflectionMask);

	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);

	assert.ok(
		resources._state.drawOptions.some(
			(entry) =>
				entry.packetId === "object" &&
				entry.sceneTargetMode === "color"
		)
	);
	const capturePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionCaptureMain"
	);
	assert.ok(capturePass);
	assert.equal(capturePass.colorAttachments.length, 1);
	assert.ok(
		resources._state.events.includes(
			"draw:mirror:mrt:planar-reflection-composite"
		)
	);
	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.ok(labels.includes("WebGPUMainColor_Clear"));
	assert.ok(labels.includes("WebGPUPlanarReflectionComposite"));
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.equal(compositePass.colorAttachments.length, 2);
	assert.strictEqual(
		compositePass.colorAttachments[1].view,
		targets.planarReflectionMask
	);
	executor.destroy();
}

async function testPlanarReflectionCaptureKeepsMSAAFrameTargetsAlive() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const msaa = createMSAAContext(4);
	let executor = null;
	const resources = createPlanarReflectionResourcesStub();
	executor = new WebGPUFrameExecutor(backend, resources, msaa);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const frameTargets = getFrameTargets(executor);
	const msaaTargets = getMSAATargets(executor);
	assert.ok(frameTargets);
	assert.ok(msaaTargets);

	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);
	assert.equal(msaa.sampleCount, 4);
	assert.strictEqual(getFrameTargets(executor), frameTargets);
	assert.strictEqual(getMSAATargets(executor), msaaTargets);

	await executor.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context
	);
	assert.strictEqual(getFrameTargets(executor), frameTargets);
	assert.strictEqual(getMSAATargets(executor), msaaTargets);
	const captureDrawOptions = resources._state.drawOptions.find(
		(options) => options.drawMode === "reflection-capture"
	);
	assert.ok(captureDrawOptions);
	assert.equal(captureDrawOptions.sceneTargetMode, "color");
	assert.equal(captureDrawOptions.sampleCount, 1);
	const compositePass = backend.recordedRenderPasses.find(
		(pass) => pass.label === "WebGPUPlanarReflectionComposite"
	);
	assert.ok(compositePass);
	assert.equal(
		compositePass.colorAttachments[0].resolveTarget,
		frameTargets.sceneColorMain
	);
	assert.equal(
		compositePass.colorAttachments[1].resolveTarget,
		frameTargets.planarReflectionMask
	);
	executor.destroy();
}

async function testPlanarReflectionCaptureFailureKeepsMainFrameResources() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	resources._state.throwOnClusteredBuild = true;
	const executor = new WebGPUFrameExecutor(backend, resources, backend.msaaContext);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});
	context.incremental = {
		enabled: false,
		forceFullFrame: true,
		dirtyRects: [{ x: 0, y: 0, width: 64, height: 64 }],
		dirtyTileSize: 64,
		dirtyTileColumns: 1,
		dirtyTileRows: 1,
		dirtyTiles: [0],
		dirtyAreaRatio: 1,
		firstPass: null,
		reasonMask: 0,
		temporalHistoryReset: false,
	};
	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const objectPacket = createPlanarPacket("object", objectMaterial, 1);
	context.scene.opaquePackets = [mirrorPacket, objectPacket];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	const mainFrameResources = executor.getPreparedFrameResources();

	await assert.rejects(
		executor.executePass(
			{ stage: "reflection", executor: "backend", enabled: true },
			context
		),
		/simulated planar capture failure/
	);

	assert.strictEqual(executor.getPreparedFrameResources(), mainFrameResources);
	assert.equal(mainFrameResources.sceneTargetMode, "mrt");
	assert.ok(
		resources._state.events.some((event) =>
			event === "scope:destroy"
		)
	);
	executor.abortFrame();
	assert.equal(getFrameGraphDebugState(executor).active, false);
}

async function testPlanarReflectionCaptureUsesMirroredCameraAndCenterSide() {
	const backend = new FakeBackend();
	backend.device.limits.maxStorageTexturesPerShaderStage = 0;
	const resources = createPlanarReflectionResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	const camera = new Camera();
	camera.position.set(0, 2, 5);
	camera.updateMatrices();
	context.viewCamera = camera;
	context.features.enableReflection = true;
	context.postProcess = createResolvedPostProcess({
		ssr: { enabled: true },
	});

	const mirrorMaterial = new Material({
		name: "mirror",
		reflectivity: 0.75,
		mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
	});
	const objectMaterial = new Material({ name: "object" });
	const mirrorPacket = createPlanarPacket("mirror", mirrorMaterial, 0);
	const abovePacket = createPlanarPacket("above", objectMaterial, 1);
	const crossingPacket = createPlanarPacket("crossing", objectMaterial, -0.5);
	const culledPacket = createPlanarPacket("culled", objectMaterial, -2);
	culledPacket.worldBounds.radius = 0.25;
	context.scene.opaquePackets = [
		mirrorPacket,
		abovePacket,
		crossingPacket,
		culledPacket,
	];
	context.scene.reflectivePackets = [mirrorPacket];
	context.scene.transparentPackets = [];
	context.scene.meshInstances = [];
	context.scene.lights = [];
	context.scene.shadowMaps = new Map();
	context.scene.environment = {
		backgroundEnabled: false,
		lightingEnabled: false,
		backgroundTexture: null,
		iblTexture: null,
		backgroundStrength: 1,
		diffuseStrength: 1,
		specularStrength: 1,
		backgroundTintLinear: { r: 1, g: 1, b: 1 },
		backgroundExposure: 1,
	};

	executor.beginFrame(context);
	await executor.executePass(
		{ stage: "reflection", executor: "backend", enabled: true },
		context
	);

	const captureContext = resources._state.prepareContexts.find(
		(candidate) =>
			candidate.features.enableReflection === false &&
			candidate.postProcess.isEnabled("ssr") === false
	);
	assert.ok(captureContext);
	assert.deepEqual(captureContext.viewCamera.getWorldPosition(), {
		x: 0,
		y: -2,
		z: 5,
	});
	assert.deepEqual(
		captureContext.scene.opaquePackets.map((packet) => packet.id),
		["above"]
	);
	assert.ok(
		resources._state.events.includes(
			"prepare:reflection:false:ssr:false:opaque:above"
		)
	);
}

async function testScreenSpaceRefractionCapturesTransmissionPackets() {
	const backend = createOITBackend();
	const resources = createOITSequencingResourcesStub();
	const executor = new WebGPUFrameExecutor(
		backend, resources, undefined, undefined, resources,
	);
	const context = createFrameContext(64, 64);
	context.postProcess = createResolvedPostProcess(
		{ ssrefraction: { enabled: true } },
		"webgpu"
	);
	context.scene.transparentPackets = [
		{
			id: "transparent-transmission-capture",
			material: { transmissionFactor: 1 },
		},
	];

	executor.beginFrame(context);
	const targets = getFrameTargets(executor);
	assert.ok(targets);
	await executor.executePass(
		{ stage: "main-transparent", executor: "backend", enabled: true },
		context
	);

	const labels = backend.recordedRenderPasses.map((pass) => pass.label);
	assert.deepEqual(labels, ["WebGPUTransmissionCapture"]);
	assert.equal(backend.encoderCopyCalls.length, 2);
	assert.equal(backend.encoderCopyCalls[0][0].texture, targets.sceneColorMain);
	assert.equal(
		backend.encoderCopyCalls[0][1].texture,
		targets.transmissionSceneColorCopy
	);
	assert.equal(backend.encoderCopyCalls[1][0].texture, targets.depth);
	assert.equal(backend.encoderCopyCalls[1][0].aspect, "depth-only");
	assert.equal(
		backend.encoderCopyCalls[1][1].texture,
		targets.transmissionDepth
	);
	assert.equal(backend.encoderCopyCalls[1][1].aspect, "depth-only");
	assert.equal(
		backend.recordedRenderPasses[0].depthStencilAttachment.depthLoadOp,
		"load"
	);
	assert.ok(
		resources._state.events.includes(
			"draw:transparent-transmission-capture:transmission-capture:default"
		)
	);
	assert.equal(
		resources._state.events.includes(
			"draw:transparent-transmission-capture:transmission:default"
		),
		false
	);
	assert.deepEqual(resources._state.drawOptions, [
		{
			packetId: "transparent-transmission-capture",
			sceneTargetMode: "mrt",
			transparentPipelineMode: "transmission-capture",
			drawMode: "default",
		},
	]);
}

async function run() {
	try {
		await testPlanarReflectionCaptureAndCompositeSequencing();
		await testPlanarReflectionUsesColorTargetsWithoutPostProcess();
		await testPlanarReflectionCaptureKeepsMSAAFrameTargetsAlive();
		await testPlanarReflectionCaptureFailureKeepsMainFrameResources();
		await testPlanarReflectionCaptureUsesMirroredCameraAndCenterSide();
		await testScreenSpaceRefractionCapturesTransmissionPackets();
		console.log("WebGPU frame-executor reflection/refraction tests passed");
	} finally {
		restoreTestState();
	}
}

await run();
