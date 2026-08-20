import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { Material } from "../../../src/materials/Material.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { AnimationClip } from "../../../src/animation/AnimationClip.ts";
import { AnimationSystem } from "../../../src/animation/AnimationSystem.ts";
import { KeyframeTrack } from "../../../src/animation/KeyframeTrack.ts";
import { AnimationRuntime } from "../../../src/simulation/animation/AnimationRuntime.ts";
import { sampleTrack } from "../../../src/simulation/animation/interpolation.ts";
import {
	ANIMATION_DEFORMATION_STATES_KEY,
	ANIMATION_RUNTIME_POSE_KEY,
	ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
} from "../../../src/simulation/animation/types.ts";

function nearlyEqual(left, right, epsilon = 1e-5) {
	return Math.abs(left - right) <= epsilon;
}

function assertNearly(left, right, message) {
	assert.ok(nearlyEqual(left, right), `${message}: ${left} !== ${right}`);
}

function createTranslationTrack(path, from, to, interpolation = "linear") {
	return new KeyframeTrack({
		binding: {
			targetType: "node",
			targetPath: path,
			property: "translation",
		},
		times: [0, 1],
		values: [...from, ...to],
		valueSize: 3,
		interpolation,
	});
}

function testTrackSampling() {
	const linearTrack = new KeyframeTrack({
		binding: {
			targetType: "node",
			targetPath: "/root/node",
			property: "translation",
		},
		times: [0, 1],
		values: [0, 0, 0, 10, 20, 30],
		valueSize: 3,
		interpolation: "linear",
	});
	const linear = sampleTrack(linearTrack, 0.25);
	assertNearly(linear[0], 2.5, "linear x");
	assertNearly(linear[1], 5, "linear y");
	assertNearly(linear[2], 7.5, "linear z");

	const stepTrack = new KeyframeTrack({
		binding: {
			targetType: "material",
			targetPath: "/material",
			property: "opacity",
		},
		times: [0, 1],
		values: [0.2, 0.9],
		valueSize: 1,
		interpolation: "step",
	});
	const stepped = sampleTrack(stepTrack, 0.75);
	assertNearly(stepped[0], 0.2, "step sample");

	const cubicTrack = new KeyframeTrack({
		binding: {
			targetType: "morph",
			targetPath: "/morph",
			property: "weights",
		},
		times: [0, 1],
		// glTF cubic: inTan, value, outTan per frame
		values: [0, 0, 0, 0, 1, 0],
		valueSize: 1,
		interpolation: "cubic",
	});
	const cubic = sampleTrack(cubicTrack, 0.5);
	assertNearly(cubic[0], 0.5, "cubic sample");

	const quaternionTrack = new KeyframeTrack({
		binding: {
			targetType: "node",
			targetPath: "/root/node",
			property: "rotation",
		},
		times: [0, 1],
		values: [0, 0, 0, 2, 0, 2, 0, 0],
		valueSize: 4,
		interpolation: "linear",
	});
	const quaternion = sampleTrack(quaternionTrack, 0.5, { isQuaternion: true });
	const length = Math.hypot(
		quaternion[0],
		quaternion[1],
		quaternion[2],
		quaternion[3]
	);
	assertNearly(length, 1, "quaternion normalization");
}

function testRuntimeLayerMaskAndAdditive() {
	const root = new Node({ name: "root" });
	const arm = new Node({ name: "arm" });
	const leg = new Node({ name: "leg" });
	root.addChild(arm);
	root.addChild(leg);

	const baseClip = new AnimationClip({
		name: "base",
		duration: 1,
		tracks: [createTranslationTrack("/root/arm", [0, 0, 0], [10, 0, 0])],
	});
	const additiveClip = new AnimationClip({
		name: "add",
		duration: 1,
		tracks: [
			createTranslationTrack("/root/arm", [1, 0, 0], [1, 0, 0]),
			createTranslationTrack("/root/leg", [7, 0, 0], [7, 0, 0]),
		],
	});

	const system = new AnimationSystem();
	const mixer = system.createMixer(root);
	mixer.addClip(baseClip);
	mixer.addClip(additiveClip);
	mixer.bindNode("/root/arm", arm);
	mixer.bindNode("/root/leg", leg);

	mixer.clipAction("base").play().setEffectiveWeight(1);
	const additiveLayer = mixer.getOrCreateLayer("Additive");
	additiveLayer.blendMode = "additive";
	additiveLayer.mask = { include: ["/root/arm"] };
	mixer.clipAction("add", "Additive").play().setEffectiveWeight(1);

	const runtime = new AnimationRuntime();
	const transient = new Map();
	runtime.update(system, 0.5, transient);

	assertNearly(arm.position.x, 6, "masked additive arm translation");
	assertNearly(leg.position.x, 0, "mask blocks leg translation");

	const poseState = transient.get(ANIMATION_RUNTIME_POSE_KEY);
	assert.ok(Array.isArray(poseState));
	assert.ok(poseState.some((entry) => entry.path === "/root/arm"));
}

function testRootMotionToggle() {
	const root = new Node({ name: "root" });
	const system = new AnimationSystem();
	const mixer = system.createMixer(root);
	mixer.bindNode("root", root);
	mixer.addClip(
		new AnimationClip({
			name: "root-motion",
			duration: 1,
			tracks: [createTranslationTrack("root", [0, 0, 0], [5, 0, 0])],
		})
	);
	const action = mixer.clipAction("root-motion").play().setEffectiveWeight(1);
	const runtime = new AnimationRuntime();

	runtime.update(system, 0.5, new Map());
	assertNearly(root.position.x, 0, "root motion disabled by default");

	mixer.rootMotion.enabled = true;
	root.position.set(0, 0, 0);
	root.updateLocalMatrix();
	action.reset().play();
	runtime.update(system, 0.5, new Map());
	assertNearly(root.position.x, 2.5, "root motion enabled translation");
}

function createMorphMesh() {
	const mesh = MeshAsset.fromFaces([
		{
			material: new Material(),
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
	mesh.primitives[0].geometry.morphTargets = [
		{
			positions: new Float32Array([
				1, 0, 0,
				1, 0, 0,
				1, 0, 0,
			]),
		},
	];
	return mesh;
}

function testDeformationRevisionBoundsAndPacketIdentity() {
	const mesh = createMorphMesh();
	const firstInstance = new MeshInstance({
		mesh,
		morphWeights: [new Float32Array([0])],
	});
	const secondInstance = new MeshInstance({
		mesh,
		morphWeights: [new Float32Array([1])],
	});
	const system = new AnimationSystem();
	const mixer = system.createMixer(new Node({ name: "root" }));
	mixer.bindMorph("/first", firstInstance);
	mixer.bindMorph("/second", secondInstance);
	const runtime = new AnimationRuntime();
	const primitiveId = mesh.primitives[0].id;
	const firstPacketId = `${firstInstance.id}:${primitiveId}`;
	const secondPacketId = `${secondInstance.id}:${primitiveId}`;

	const firstTransient = new Map();
	runtime.resolveDeformations(system, firstTransient);
	const firstStates = firstTransient.get(ANIMATION_DEFORMATION_STATES_KEY);
	assert.equal(firstStates.size, 2);
	assert.ok(firstStates.has(firstPacketId));
	assert.ok(firstStates.has(secondPacketId));
	assert.ok(
		firstStates.get(secondPacketId).localBounds.center.x >
			firstStates.get(firstPacketId).localBounds.center.x + 0.9
	);
	assert.ok(
		firstTransient
			.get(ANIMATION_SOFTWARE_DEFORMED_GEOMETRY_KEY)
			.has(firstPacketId)
	);
	assert.ok(
		firstTransient
			.get(ANIMATION_MORPH_WEIGHTS_KEY)
			.has(secondPacketId)
	);

	const unchangedTransient = new Map();
	runtime.resolveDeformations(system, unchangedTransient);
	const unchangedStates = unchangedTransient.get(ANIMATION_DEFORMATION_STATES_KEY);
	assert.equal(
		unchangedStates.get(firstPacketId).revision,
		firstStates.get(firstPacketId).revision
	);

	firstInstance.morphWeights[0][0] = 0.5;
	const changedTransient = new Map();
	runtime.resolveDeformations(system, changedTransient);
	const changedState = changedTransient
		.get(ANIMATION_DEFORMATION_STATES_KEY)
		.get(firstPacketId);
	assert.notEqual(changedState.revision, firstStates.get(firstPacketId).revision);
	assert.ok(
		changedState.localBounds.center.x >
			firstStates.get(firstPacketId).localBounds.center.x
	);
}

function testSceneDeformationWithoutMixerBinding() {
	const scene = new Scene();
	const mesh = createMorphMesh();
	const instance = scene.add(new MeshInstance({
		mesh,
		morphWeights: [new Float32Array([0.5])],
	}));
	const runtime = new AnimationRuntime();
	const transient = new Map();
	runtime.resolveDeformations(new AnimationSystem(), transient, scene);
	const packetId = `${instance.id}:${mesh.primitives[0].id}`;
	assert.ok(transient.get(ANIMATION_DEFORMATION_STATES_KEY).has(packetId));
	assert.ok(transient.get(ANIMATION_MORPH_WEIGHTS_KEY).has(packetId));
}

function run() {
	testTrackSampling();
	testRuntimeLayerMaskAndAdditive();
	testRootMotionToggle();
	testDeformationRevisionBoundsAndPacketIdentity();
	testSceneDeformationWithoutMixerBinding();
	console.log("Animation core tests passed");
}

run();
