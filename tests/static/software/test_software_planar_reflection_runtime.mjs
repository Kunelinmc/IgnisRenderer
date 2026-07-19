import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Camera } from "../../../src/cameras/Camera.ts";
import { BasicMaterial, AlphaMode } from "../../../src/materials/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { Rasterizer } from "../../../src/backends/software/Rasterizer.ts";
import { SoftwareMainPass } from "../../../src/backends/software/passes/SoftwareMainPass.ts";
import {
	SoftwareReflectionPass,
} from "../../../src/backends/software/passes/SoftwareReflectionPass.ts";
import {
	getSoftwarePlanarReflectionRuntime,
	resolveSoftwarePlanarReflectionPlaneKey,
	SoftwarePlanarReflectionRuntime,
} from "../../../src/backends/software/SoftwarePlanarReflectionRuntime.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const WIDTH = 64;
const HEIGHT = 64;

if (typeof globalThis.ImageData !== "function") {
	globalThis.ImageData = class ImageData {
		constructor(dataOrWidth, width, height) {
			if (dataOrWidth instanceof Uint8ClampedArray) {
				this.data = dataOrWidth;
				this.width = width;
				this.height = height;
			} else {
				this.width = dataOrWidth;
				this.height = width;
				this.data = new Uint8ClampedArray(this.width * this.height * 4);
			}
		}
	};
}

function createZeroSH() {
	return Array.from({ length: 9 }, () => ({ r: 0, g: 0, b: 0 }));
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

function createContext(camera, packetsByStage = {}, features = {}) {
	const zeroSH = createZeroSH();
	return {
		viewCamera: camera,
		attachments: {
			pixels: new Uint8ClampedArray(WIDTH * HEIGHT * 4),
			depthBuffer: new Float32Array(WIDTH * HEIGHT).fill(Infinity),
			normalBuffer: null,
			motionBuffer: null,
			width: WIDTH,
			height: HEIGHT,
		},
		features: {
			enableLighting: false,
			enableSH: false,
			enableShadows: false,
			enableReflection: features.enableReflection ?? true,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess({
			gamma: { enabled: false },
			tonemap: { enabled: false },
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
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: packetsByStage.reflectivePackets ?? [],
			decalPackets: [],
			spatialIndex: null,
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
	const material = new BasicMaterial({
		name: `material-${id}`,
		diffuse: color,
		doubleSided: true,
		alphaMode: options.alphaMode ?? AlphaMode.Opaque,
		opacity: options.opacity ?? 1,
	});
	material.reflectivity = options.reflectivity ?? material.reflectivity;
	material.mirrorPlane = options.mirrorPlane ?? material.mirrorPlane;

	const geometry = {
		positions: new Float32Array([
			-0.8,
			-0.8,
			0,
			0.8,
			-0.8,
			0,
			0.0,
			0.8,
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
	return {
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
	};
}

function createImageBuffer(width, height, data) {
	return {
		imageData: new ImageData(new Uint8ClampedArray(data), width, height),
		width,
		height,
	};
}

function createAlphaTexture(alpha) {
	return {
		data: new Uint8ClampedArray([255, 255, 255, alpha]),
		width: 1,
		height: 1,
		repeat: { x: 1, y: 1 },
		offset: { x: 0, y: 0 },
		rotation: 0,
		wrapS: "ClampToEdge",
		wrapT: "ClampToEdge",
	};
}

function testReflectionPassPublishesRuntime() {
	const pass = new SoftwareReflectionPass(new Rasterizer());
	const context = createContext(createCamera(), {}, { enableReflection: true });

	pass.render(context);

	assert.ok(
		getSoftwarePlanarReflectionRuntime(context.transient),
		"Reflection pass should publish its runtime through transient storage"
	);
}

async function testRuntimeCompositeBlendsReflectionColor() {
	const rasterizer = new Rasterizer();
	const mainPass = new SoftwareMainPass(rasterizer, {
		mode: "scanline",
		enableEarlyZPrepass: false,
	});
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const packet = createTrianglePacket(
		"mirror",
		{ r: 200, g: 0, b: 0 },
		{ reflectivity: 0.5, mirrorPlane }
	);
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [packet],
			reflectivePackets: [packet],
		},
		{ enableReflection: true }
	);
	const runtime = new SoftwarePlanarReflectionRuntime(rasterizer);
	runtime.reflectionBuffers.set(
		resolveSoftwarePlanarReflectionPlaneKey(mirrorPlane),
		createImageBuffer(1, 1, [0, 200, 0, 255])
	);

	await mainPass.render(context, [packet], false);
	runtime.composite(context, [packet]);

	const idx = ((HEIGHT >> 1) * WIDTH + (WIDTH >> 1)) << 2;
	assert.ok(
		context.attachments.pixels[idx] > 80 &&
			context.attachments.pixels[idx] < 120,
		"Main pass should composite reflection red channel"
	);
	assert.ok(
		context.attachments.pixels[idx + 1] > 80 &&
			context.attachments.pixels[idx + 1] < 120,
		"Main pass should composite reflection green channel"
	);
}

async function testRuntimeCompositeSamplesScaledReflectionBuffer() {
	const rasterizer = new Rasterizer();
	const mainPass = new SoftwareMainPass(rasterizer, {
		mode: "scanline",
		enableEarlyZPrepass: false,
	});
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const packet = createTrianglePacket(
		"mirror-scaled",
		{ r: 0, g: 0, b: 0 },
		{ reflectivity: 1, mirrorPlane }
	);
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [packet],
			reflectivePackets: [packet],
		},
		{ enableReflection: true }
	);
	const runtime = new SoftwarePlanarReflectionRuntime(rasterizer);
	runtime.reflectionBuffers.set(
		resolveSoftwarePlanarReflectionPlaneKey(mirrorPlane),
		createImageBuffer(2, 2, [
			10,
			0,
			0,
			255,
			20,
			0,
			0,
			255,
			30,
			0,
			0,
			255,
			40,
			0,
			0,
			255,
		])
	);

	await mainPass.render(context, [packet], false);
	runtime.composite(context, [packet]);

	const idx = ((HEIGHT >> 1) * WIDTH + (WIDTH >> 1)) << 2;
	assert.equal(context.attachments.pixels[idx], 40);
}

function testRuntimeCompositeSkipsMissingBuffer() {
	const rasterizer = new Rasterizer();
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const packet = createTrianglePacket(
		"mirror-missing-buffer",
		{ r: 0, g: 0, b: 0 },
		{ reflectivity: 1, mirrorPlane }
	);
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [packet],
			reflectivePackets: [packet],
		},
		{ enableReflection: true }
	);
	context.attachments.pixels.fill(100);
	const runtime = new SoftwarePlanarReflectionRuntime(rasterizer);

	runtime.composite(context, [packet]);

	const idx = ((HEIGHT >> 1) * WIDTH + (WIDTH >> 1)) << 2;
	assert.equal(context.attachments.pixels[idx], 100);
}

function testRuntimeCompositeHonorsAlphaMask() {
	const rasterizer = new Rasterizer();
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const packet = createTrianglePacket(
		"mirror-mask",
		{ r: 0, g: 0, b: 0 },
		{
			reflectivity: 1,
			mirrorPlane,
			alphaMode: AlphaMode.Mask,
		}
	);
	packet.material.map = createAlphaTexture(0);
	packet.material.alphaCutoff = 0.5;
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [packet],
			reflectivePackets: [packet],
		},
		{ enableReflection: true }
	);
	context.attachments.pixels.fill(100);
	const runtime = new SoftwarePlanarReflectionRuntime(rasterizer);
	runtime.reflectionBuffers.set(
		resolveSoftwarePlanarReflectionPlaneKey(mirrorPlane),
		createImageBuffer(1, 1, [0, 200, 0, 255])
	);

	runtime.composite(context, [packet]);

	const idx = ((HEIGHT >> 1) * WIDTH + (WIDTH >> 1)) << 2;
	assert.equal(context.attachments.pixels[idx], 100);
}

async function testReflectionPassCompositeAfterMainPass() {
	const rasterizer = new Rasterizer();
	const mainPass = new SoftwareMainPass(rasterizer, {
		mode: "scanline",
		enableEarlyZPrepass: false,
	});
	const reflectionPass = new SoftwareReflectionPass(rasterizer);
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const packet = createTrianglePacket(
		"mirror-pass",
		{ r: 200, g: 0, b: 0 },
		{ reflectivity: 0.5, mirrorPlane }
	);
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [packet],
			reflectivePackets: [packet],
		},
		{ enableReflection: true }
	);

	await mainPass.render(context, [packet], false);
	reflectionPass.render(context);
	const runtime = getSoftwarePlanarReflectionRuntime(context.transient);
	assert.ok(runtime, "Reflection pass should expose the runtime for the frame");
	runtime.reflectionBuffers.set(
		resolveSoftwarePlanarReflectionPlaneKey(mirrorPlane),
		createImageBuffer(1, 1, [0, 200, 0, 255])
	);
	reflectionPass.composite(context, [packet]);

	const idx = ((HEIGHT >> 1) * WIDTH + (WIDTH >> 1)) << 2;
	assert.ok(
		context.attachments.pixels[idx] > 80 &&
			context.attachments.pixels[idx] < 120,
		"Composite pass should blend reflection red channel"
	);
	assert.ok(
		context.attachments.pixels[idx + 1] > 80 &&
			context.attachments.pixels[idx + 1] < 120,
		"Composite pass should blend reflection green channel"
	);
}

function testReflectionCaptureDisablesRecursiveComposite() {
	const drawContexts = [];
	const fakeRasterizer = {
		drawTriangle(_pts, _face, _pixels, context) {
			drawContexts.push(context);
		},
	};
	const mirrorPlane = { normal: { x: 0, y: 0, z: 1 }, constant: 0 };
	const mirrorPacket = createTrianglePacket(
		"mirror",
		{ r: 100, g: 100, b: 100 },
		{ reflectivity: 0.5, mirrorPlane }
	);
	const reflectedPacket = createTrianglePacket(
		"reflected",
		{ r: 40, g: 80, b: 120 },
		{ zOffset: 1 }
	);
	const context = createContext(
		createCamera(),
		{
			opaquePackets: [mirrorPacket, reflectedPacket],
			reflectivePackets: [mirrorPacket],
		},
		{ enableReflection: true }
	);
	const runtime = new SoftwarePlanarReflectionRuntime(fakeRasterizer);

	runtime.render(context);

	assert.ok(
		drawContexts.length > 0,
		"Reflection capture should rasterize reflected scene geometry"
	);
	for (const drawContext of drawContexts) {
		assert.equal("enableReflection" in drawContext, false);
		assert.equal("planarReflectionComposite" in drawContext, false);
	}
}

function testRasterizerDoesNotReferencePlanarReflection() {
	const source = readFileSync(
		new URL("../../../src/backends/software/Rasterizer.ts", import.meta.url),
		"utf8"
	);
	assert.equal(/planarReflection|SoftwarePlanarReflection/.test(source), false);
}

async function run() {
	testReflectionPassPublishesRuntime();
	await testRuntimeCompositeBlendsReflectionColor();
	await testRuntimeCompositeSamplesScaledReflectionBuffer();
	testRuntimeCompositeSkipsMissingBuffer();
	testRuntimeCompositeHonorsAlphaMask();
	await testReflectionPassCompositeAfterMainPass();
	testReflectionCaptureDisablesRecursiveComposite();
	testRasterizerDoesNotReferencePlanarReflection();
	console.log("Software planar reflection runtime tests passed");
}

run();
