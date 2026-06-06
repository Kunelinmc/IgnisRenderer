import assert from "node:assert/strict";
import { Camera } from "../../../src/cameras/Camera.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Decal } from "../../../src/decals/Decal.ts";
import { Material } from "../../../src/materials/Material.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { PreparedSceneBuilder } from "../../../src/pipeline/PreparedSceneBuilder.ts";

function createTriangleMesh(material) {
	return MeshAsset.fromFaces([
		{
			material,
			vertices: [
				{
					x: 0,
					y: 0,
					z: 0,
					u: 0,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 1,
					y: 0,
					z: 0,
					u: 1,
					v: 0,
					normal: { x: 0, y: 0, z: 1 },
				},
				{
					x: 0,
					y: 1,
					z: 0,
					u: 0,
					v: 1,
					normal: { x: 0, y: 0, z: 1 },
				},
			],
		},
	]);
}

function createRenderer(scene, camera) {
	return {
		scene,
		camera,
		shadowMaps: new Map(),
		animationSystem: {
			hasActiveActions() {
				return false;
			},
		},
	};
}

function testPreparedSceneDecalPackets() {
	const scene = new Scene();
	const camera = new Camera();
	camera.position.set(0, 0, 5);
	scene.add(camera);

	const receiverMaterial = new Material({
		name: "Receiver",
		alphaMode: "OPAQUE",
	});
	const transparentMaterial = new Material({
		name: "TransparentReceiver",
		alphaMode: "BLEND",
	});
	const receiver = scene.add(
		new MeshInstance({
			mesh: createTriangleMesh(receiverMaterial),
			name: "receiver",
		})
	);
	receiver.renderLayers = 4.8;
	assert.equal(receiver.renderLayers, 4);

	const transparentReceiver = scene.add(
		new MeshInstance({
			mesh: createTriangleMesh(transparentMaterial),
			name: "transparentReceiver",
		})
	);
	transparentReceiver.renderLayers = 8;

	const firstMaterial = new Material({ name: "FirstDecal" });
	const secondMaterial = new Material({ name: "SecondDecal" });
	const highMaterial = new Material({ name: "HighDecal" });
	const high = scene.add(
		new Decal({
			name: "high",
			material: highMaterial,
			receiverLayerMask: 4,
			priority: 5,
			opacity: 0.6,
			edgeFade: 0.25,
			channelBlendModes: {
				normal: "disabled",
				roughness: "replace",
			},
		})
	);
	const first = scene.add(
		new Decal({
			name: "first",
			material: firstMaterial,
			receiverLayerMask: 4,
			priority: 1,
		})
	);
	const mismatch = scene.add(
		new Decal({
			name: "mismatch",
			material: new Material({ name: "MismatchDecal" }),
			receiverLayerMask: 2,
			priority: 0,
		})
	);
	const second = scene.add(
		new Decal({
			name: "second",
			material: secondMaterial,
			receiverLayerMask: 4,
			priority: 1,
		})
	);
	const transparentOnly = scene.add(
		new Decal({
			name: "transparentOnly",
			material: new Material({ name: "TransparentOnlyDecal" }),
			receiverLayerMask: 8,
		})
	);
	const shaderMaterial = scene.add(
		new Decal({
			name: "shaderMaterial",
			material: new ShaderMaterial({ name: "CustomDecalShader" }),
			receiverLayerMask: 4,
		})
	);
	const disabled = scene.add(
		new Decal({
			name: "disabled",
			material: new Material({ name: "DisabledDecal" }),
			receiverLayerMask: 0,
		})
	);
	const hidden = scene.add(
		new Decal({
			name: "hidden",
			material: new Material({ name: "HiddenDecal" }),
			receiverLayerMask: 4,
			opacity: 0,
		})
	);

	scene.updateWorldMatrices();
	camera.updateMatrices();

	assert.deepEqual(
		scene.getDecals().map((decal) => decal.name),
		[
			"high",
			"first",
			"mismatch",
			"second",
			"transparentOnly",
			"shaderMaterial",
			"disabled",
			"hidden",
		]
	);

	const frame = PreparedSceneBuilder.build(createRenderer(scene, camera));
	assert.equal(frame.opaquePackets.length, 1);
	assert.equal(frame.transparentPackets.length, 1);
	assert.deepEqual(
		frame.decalPackets.map((packet) => packet.decal.name),
		["first", "second", "high"]
	);
	assert.equal(frame.decalPackets[0].material, firstMaterial);
	assert.equal(frame.decalPackets[1].material, secondMaterial);

	const highPacket = frame.decalPackets[2];
	assert.equal(highPacket.decal, high);
	assert.equal(highPacket.material, highMaterial);
	assert.equal(highPacket.receiverLayerMask, 4);
	assert.equal(highPacket.opacity, 0.6);
	assert.equal(highPacket.edgeFade, 0.25);
	assert.equal(highPacket.channelBlendModes.normal, "disabled");
	assert.equal(highPacket.channelBlendModes.roughness, "replace");
	assert.ok(highPacket.worldBounds.radius > 0);
	assert.notEqual(highPacket.inverseWorldMatrix, highPacket.worldMatrix);

	for (const skipped of [
		mismatch,
		transparentOnly,
		shaderMaterial,
		disabled,
		hidden,
	]) {
		assert.equal(
			frame.decalPackets.some((packet) => packet.decal === skipped),
			false
		);
	}
}

function testDecalClonePreservesDecalProperties() {
	const material = new Material({ name: "CloneSource" });
	const decal = new Decal({
		name: "cloneSource",
		material,
		receiverLayerMask: 16,
		priority: 3,
		opacity: 0.5,
		edgeFade: 0.2,
		channelBlendModes: {
			baseColor: "multiply",
			normal: "normal",
		},
	});
	decal.renderLayers = 32;
	decal.position.set(1, 2, 3);

	const clone = decal.clone(false);
	assert.equal(clone.name, "cloneSource");
	assert.equal(clone.material, material);
	assert.equal(clone.receiverLayerMask, 16);
	assert.equal(clone.priority, 3);
	assert.equal(clone.opacity, 0.5);
	assert.equal(clone.edgeFade, 0.2);
	assert.equal(clone.renderLayers, 32);
	assert.equal(clone.channelBlendModes.baseColor, "multiply");
	assert.equal(clone.channelBlendModes.normal, "normal");
	assert.notEqual(clone.channelBlendModes, decal.channelBlendModes);
}

testPreparedSceneDecalPackets();
testDecalClonePreservesDecalProperties();

console.log("Prepared scene decal tests passed");
