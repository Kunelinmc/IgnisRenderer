import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { BasicMaterial, AlphaMode, PBRMaterial } from "../../../src/materials/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { Projector } from "../../../src/backends/software/Projector.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

const WIDTH = 64;
const HEIGHT = 64;

function createZeroSH() {
	return Array.from({ length: 9 }, () => ({ r: 0, g: 0, b: 0 }));
}

function createRendererBridge() {
	return {
		canvas: { width: WIDTH, height: HEIGHT },
	};
}

function createSoftwareSession(options = {}) {
	const backend = new SoftwareBackend(options);
	backend.attach({
		surface: createRendererBridge(),
		events: { emit: () => {} },
	});
	return backend;
}

function createContext(backend, camera, packetsByStage = {}) {
	const attachments = backend.getAttachments({ width: WIDTH, height: HEIGHT });
	const zeroSH = createZeroSH();

	return {
		presentationAlphaMode: packetsByStage.presentationAlphaMode ?? "opaque",
		viewCamera: camera,
		attachments,
		features: {
			enableLighting: false,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess({
			gamma: { enabled: false },
		}),
		shadowMaps: new Map(),
		scene: {
			sceneBounds: {
				center: { x: 0, y: 0, z: 0 },
				radius: 100,
			},
			lights: [],
			particleSystems: [],
			hasActiveAnimations: false,
			camera,
			environment: null,
			meshInstances: [],
			shadowMaps: new Map(),
			opaquePackets: packetsByStage.opaquePackets ?? [],
			transparentPackets: packetsByStage.transparentPackets ?? [],
			shadowCasterSubmissions: [],
			shadowTransmitterSubmissions: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: zeroSH,
		shAmbientCoeffs: zeroSH,
		worldMatrix: Matrix4.identity(),
		incremental: {
			enabled: false,
			forceFullFrame: true,
			dirtyRects: [],
			dirtyTileSize: 0,
			dirtyTileColumns: 0,
			dirtyTileRows: 0,
			dirtyTiles: [],
			dirtyAreaRatio: 1,
			firstPass: null,
			reasonMask: 0,
			temporalHistoryReset: true,
		},
		transient: new Map(),
	};
}

function createTrianglePacket(id, color, options = {}) {
	const zOffset = options.zOffset ?? 0;
	const alphaMode = options.alphaMode ?? AlphaMode.Opaque;
	const opacity = options.opacity ?? 1;
	const material = new BasicMaterial({
		name: `material-${id}`,
		diffuse: color,
		doubleSided: true,
		alphaMode,
		opacity,
	});

	const geometry = {
		positions: new Float32Array([
			-0.6,
			-0.6,
			0,
			0.6,
			-0.6,
			0,
			0.0,
			0.6,
			0,
		]),
		normals: new Float32Array([
			0,
			0,
			1,
			0,
			0,
			1,
			0,
			0,
			1,
		]),
		indices: new Uint32Array([0, 1, 2]),
	};

	const primitive = {
		id: `primitive-${id}`,
		geometry,
		material,
		boundingSphere: {
			center: { x: 0, y: 0, z: 0 },
			radius: 2,
		},
		boundingBox: {
			min: { x: -1, y: -1, z: -1 },
			max: { x: 1, y: 1, z: 1 },
		},
		visible: true,
		castShadows: false,
		receiveShadows: false,
	};

	const worldMatrix = Matrix4.fromTranslation([0, 0, zOffset]);
	const mesh = {
		boundingSphere: {
			center: { x: 0, y: 0, z: 0 },
			radius: 2,
		},
		primitives: [primitive],
	};
	const meshInstance = {
		id: `mesh-instance-${id}`,
		mesh,
		worldMatrix,
		skeleton: null,
		visible: true,
	};

	return createTestDrawPacket({
		id: `packet-${id}`,
		meshInstance,
		mesh,
		primitive,
		material,
		geometry,
		worldMatrix,
		normalMatrix: Matrix4.identity(),
		worldBounds: {
			center: { x: 0, y: 0, z: zOffset },
			radius: 2,
		},
		sortDepth: 0,
		pipelineKey: `packet-${id}`,
		passFlags: 0,
	});
}

function createCamera() {
	const camera = new Camera();
	camera.position.set(0, 0, 4);
	camera.aspectRatio = WIDTH / HEIGHT;
	camera.near = 0.1;
	camera.far = 100;
	camera.updateMatrices();
	return camera;
}

async function renderPass(backend, stage, packetsByStage) {
	await backend.initialize();
	const camera = createCamera();
	const context = createContext(backend, camera, packetsByStage);
	backend.beginFrame(context);
	await backend.executePass(
		{
			stage,
			executor: "backend",
			enabled: true,
		},
		context
	);
	backend.endFrame();
	return context.attachments;
}

async function testOpaqueScanlineDepthOrdering() {
	const backend = createSoftwareSession();
	const attachments = await renderPass(backend, "main-opaque", {
		opaquePackets: [
			createTrianglePacket("far-blue", { r: 0, g: 0, b: 255 }, {
				zOffset: -0.2,
			}),
			createTrianglePacket("near-red", { r: 255, g: 0, b: 0 }, {
				zOffset: 0.2,
			}),
		],
	});
	const centerPixel = (Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
	assert.ok(attachments.pixels[centerPixel] > attachments.pixels[centerPixel + 2]);
	assert.ok(Number.isFinite(attachments.depthBuffer[centerPixel / 4]));
	backend.destroy();
}

async function testRasterPreservesHDRRadiance() {
	const backend = createSoftwareSession();
	await backend.initialize();
	const camera = createCamera();
	const packet = createTrianglePacket("hdr-emissive", { r: 0, g: 0, b: 0 });
	const material = new PBRMaterial({
		albedo: { r: 0, g: 0, b: 0 },
		emissive: { r: 255, g: 0, b: 0 },
		emissiveIntensity: 4,
		doubleSided: true,
	});
	packet.submission.material.effective = material;
	const context = createContext(backend, camera, { opaquePackets: [packet] });
	context.features.enableLighting = true;
	backend.beginFrame(context);
	await backend.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);
	backend.endFrame();
	const centerPixel =
		(Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
	assert.ok(backend._surface.getSceneColorTarget()[centerPixel] >= 4);
	backend.destroy();
}

async function testTransparentRasterPreservesHDRRadiance() {
	const backend = createSoftwareSession();
	await backend.initialize();
	const camera = createCamera();
	const packet = createTrianglePacket("hdr-transparent", { r: 0, g: 0, b: 0 });
	const material = new PBRMaterial({
		albedo: { r: 0, g: 0, b: 0 },
		emissive: { r: 255, g: 0, b: 0 },
		emissiveIntensity: 4,
		alphaMode: AlphaMode.Blend,
		opacity: 1,
		doubleSided: true,
	});
	packet.submission.material.effective = material;
	const context = createContext(backend, camera, { transparentPackets: [packet] });
	context.features.enableLighting = true;
	backend.beginFrame(context);
	await backend.executePass(
		{ stage: "main-transparent", executor: "backend", enabled: true },
		context,
	);
	backend.endFrame();
	const centerPixel =
		(Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
	assert.ok(backend._surface.getSceneColorTarget()[centerPixel] >= 4);
	backend.destroy();
}

async function testIncrementalRasterClipsOutsideDirtyRegion() {
	const backend = createSoftwareSession();
	await backend.initialize();
	const camera = createCamera();
	const context = createContext(backend, camera, {
		opaquePackets: [createTrianglePacket("incremental", { r: 255, g: 0, b: 0 })],
	});
	context.incremental.enabled = true;
	context.incremental.forceFullFrame = false;
	context.incremental.dirtyRects = [{ x: 30, y: 30, width: 4, height: 4 }];
	backend.beginFrame(context);
	await backend.executePass(
		{ stage: "main-opaque", executor: "backend", enabled: true },
		context,
	);
	backend.endFrame();

	let dirtyPixels = 0;
	let outsidePixels = 0;
	for (let y = 0; y < HEIGHT; y++) {
		for (let x = 0; x < WIDTH; x++) {
			const pixel = (y * WIDTH + x) * 4;
			const lit = context.attachments.pixels[pixel] > 0;
			if (x >= 30 && x < 34 && y >= 30 && y < 34) {
				if (lit) dirtyPixels++;
			} else if (lit) {
				outsidePixels++;
			}
		}
	}
	assert.ok(dirtyPixels > 0, "incremental dirty region should be rasterized");
	assert.equal(outsidePixels, 0, "incremental raster must not write outside dirty regions");
	backend.destroy();
}

async function testTransparentScanlinePreservesPacketOrder() {
	const redPacket = createTrianglePacket(
		"transparent-red",
		{ r: 255, g: 0, b: 0 },
		{ alphaMode: AlphaMode.Blend, opacity: 0.65 }
	);
	const greenPacket = createTrianglePacket(
		"transparent-green",
		{ r: 0, g: 255, b: 0 },
		{ alphaMode: AlphaMode.Blend, opacity: 0.65 }
	);
	const backend = createSoftwareSession();
	const attachments = await renderPass(backend, "main-transparent", {
		transparentPackets: [redPacket, greenPacket],
	});
	const centerPixel = (Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
	assert.ok(attachments.pixels[centerPixel + 1] > attachments.pixels[centerPixel]);
	backend.destroy();
}

async function testTransparentPresentationCoverage() {
	for (const textureAlpha of [null, 128]) {
		const packet = createTrianglePacket(
			"transparent-coverage",
			{ r: 255, g: 0, b: 0 },
			{ alphaMode: AlphaMode.Blend, opacity: 0.5 },
		);
		if (textureAlpha !== null) {
			packet.submission.material.effective.map = {
				data: new Uint8ClampedArray([255, 255, 255, textureAlpha]),
				width: 1,
				height: 1,
				repeat: { x: 1, y: 1 },
				offset: { x: 0, y: 0 },
				rotation: 0,
				wrapS: "ClampToEdge",
				wrapT: "ClampToEdge",
			};
		}
		const backend = createSoftwareSession();
		const attachments = await renderPass(backend, "main-transparent", {
			presentationAlphaMode: "premultiplied",
			transparentPackets: [packet],
		});
		const centerPixel =
			(Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
		const sceneColor = backend._surface.getSceneColorTarget();
		const expectedAlpha = 0.5 * (textureAlpha === null ? 1 : textureAlpha / 255);
		assert.ok(Math.abs(sceneColor[centerPixel] - expectedAlpha) < 1e-6);
		assert.ok(
			Math.abs(sceneColor[centerPixel + 3] - expectedAlpha) < 1e-6,
			"material opacity and texture alpha must each contribute exactly once",
		);
		assert.ok(Math.abs(attachments.pixels[centerPixel + 3] - expectedAlpha * 255) <= 1);
		assert.equal(attachments.pixels[3], 0, "transparent clear must retain zero alpha");
		backend.destroy();
	}
}

async function testTransparentWireframeUsesMaterialOpacity() {
	const packet = createTrianglePacket(
		"transparent-wireframe",
		{ r: 255, g: 0, b: 0 },
		{ alphaMode: AlphaMode.Blend, opacity: 0.35 },
	);
	packet.submission.material.effective.wireframe = true;
	const backend = createSoftwareSession();
	await renderPass(backend, "main-transparent", {
		presentationAlphaMode: "premultiplied",
		transparentPackets: [packet],
	});
	const sceneColor = backend._surface.getSceneColorTarget();
	let wireframePixels = 0;
	for (let pixel = 0; pixel < sceneColor.length; pixel += 4) {
		if (sceneColor[pixel + 1] <= 0) continue;
		wireframePixels++;
		assert.ok(
			Math.abs(sceneColor[pixel + 3] - 0.35) < 1e-6,
			"wireframe coverage must use material opacity",
		);
	}
	assert.ok(wireframePixels > 0, "transparent wireframe edges must be rendered");
	backend.destroy();
}

async function testOpaqueMaterialWritesFullCoverageOnTransparentSurface() {
	const backend = createSoftwareSession();
	const attachments = await renderPass(backend, "main-opaque", {
		presentationAlphaMode: "premultiplied",
		opaquePackets: [createTrianglePacket(
			"opaque-coverage",
			{ r: 0, g: 0, b: 255 },
			{ opacity: 0.2 },
		)],
	});
	const centerPixel =
		(Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2)) * 4;
	assert.equal(attachments.pixels[centerPixel + 3], 255);
	backend.destroy();
}

async function testMaterialDoubleSidedControlsCulling() {
	for (const backFacing of [false, true]) {
		const packet = createTrianglePacket(
			"material-culling",
			{ r: 255, g: 0, b: 0 },
		);
		if (backFacing) {
			packet.submission.geometry.data.indices.set([0, 2, 1]);
		}
		for (const doubleSided of [false, true]) {
			packet.submission.material.effective.doubleSided = doubleSided;
			const backend = createSoftwareSession();
			const attachments = await renderPass(backend, "main-opaque", {
				opaquePackets: [packet],
			});
			const center = Math.floor(HEIGHT / 2) * WIDTH + Math.floor(WIDTH / 2);
			const shouldRender = !backFacing || doubleSided;
			const message = `backFacing=${backFacing}, material.doubleSided=${doubleSided}`;
			assert.equal(attachments.pixels[center * 4] > 0, shouldRender, message);
			assert.equal(Number.isFinite(attachments.depthBuffer[center]), shouldRender, message);
			backend.destroy();
		}
	}
}

function testMissingActiveDeformationSkipsSoftwarePacket() {
	const packet = createTrianglePacket(
		"missing-deformation",
		{ r: 255, g: 255, b: 255 },
	);
	Object.assign(packet.submission.deformation, {
		mode: "skin",
		revision: 1,
		jointPayloadKey: "missing-instance",
	});
	assert.deepEqual(
		Projector.getPacketFacesWithFrame(packet, {
			animationDeformedGeometry: null,
		}),
		[],
	);
}

async function run() {
	await testOpaqueScanlineDepthOrdering();
	await testRasterPreservesHDRRadiance();
	await testTransparentRasterPreservesHDRRadiance();
	await testIncrementalRasterClipsOutsideDirtyRegion();
	await testTransparentScanlinePreservesPacketOrder();
	await testTransparentPresentationCoverage();
	await testTransparentWireframeUsesMaterialOpacity();
	await testOpaqueMaterialWritesFullCoverageOnTransparentSurface();
	await testMaterialDoubleSidedControlsCulling();
	testMissingActiveDeformationSkipsSoftwarePacket();
	console.log("Software scanline raster tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
