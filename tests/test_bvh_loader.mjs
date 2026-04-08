import assert from "node:assert/strict";
import { BVHLoader } from "../src/loaders/BVHLoader.ts";
import { Scene } from "../src/core/Scene.ts";

const SAMPLE_BVH = `HIERARCHY
ROOT Hips
{
	OFFSET 0 0 0
	CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
	JOINT Chest
	{
		OFFSET 0 10 0
		CHANNELS 3 Zrotation Xrotation Yrotation
		End Site
		{
			OFFSET 0 10 0
		}
	}
}
MOTION
Frames: 2
Frame Time: 0.5
0 0 0 0 0 0 0 0 0
10 5 -2 90 0 0 0 45 0`;

function nearlyEqual(left, right, epsilon = 1e-5) {
	return Math.abs(left - right) <= epsilon;
}

function assertNearly(left, right, message) {
	assert.ok(nearlyEqual(left, right), `${message}: ${left} !== ${right}`);
}

function testParseHierarchyAndTracks() {
	const loader = new BVHLoader();
	const root = loader.parse(SAMPLE_BVH, { clipName: "sampleMotion" });
	const bundle = loader.getLastAnimationBundle();

	assert.equal(root.name, "bvhRoot");
	assert.equal(root.children.length, 1);
	assert.equal(root.children[0].name, "Hips");
	assert.equal(root.children[0].children.length, 1);
	assert.equal(root.children[0].children[0].name, "Chest");

	assert.ok(bundle);
	assert.equal(bundle.clips.length, 1);
	assert.equal(bundle.clips[0].name, "sampleMotion");
	assert.equal(bundle.clips[0].duration, 0.5);
	assert.equal(bundle.clips[0].tracks.length, 3);

	const rootPath = Object.keys(bundle.nodePathMap).find((path) =>
		path.endsWith("/Hips")
	);
	const chestPath = Object.keys(bundle.nodePathMap).find((path) =>
		path.endsWith("/Chest")
	);
	assert.ok(rootPath);
	assert.ok(chestPath);

	const rootTranslationTrack = bundle.clips[0].tracks.find(
		(track) =>
			track.binding.targetPath === rootPath &&
			track.binding.property === "translation"
	);
	assert.ok(rootTranslationTrack);
	assert.deepEqual(Array.from(rootTranslationTrack.times), [0, 0.5]);
	assert.deepEqual(Array.from(rootTranslationTrack.values), [0, 0, 0, 10, 5, -2]);

	const rootRotationTrack = bundle.clips[0].tracks.find(
		(track) =>
			track.binding.targetPath === rootPath &&
			track.binding.property === "rotation"
	);
	assert.ok(rootRotationTrack);
	assertNearly(rootRotationTrack.values[4], 0, "root rotation x");
	assertNearly(rootRotationTrack.values[5], 0, "root rotation y");
	assertNearly(Math.abs(rootRotationTrack.values[6]), Math.SQRT1_2, "root rotation z");
	assertNearly(Math.abs(rootRotationTrack.values[7]), Math.SQRT1_2, "root rotation w");

	const chestRotationTrack = bundle.clips[0].tracks.find(
		(track) =>
			track.binding.targetPath === chestPath &&
			track.binding.property === "rotation"
	);
	assert.ok(chestRotationTrack);
	assertNearly(
		Math.abs(chestRotationTrack.values[4]),
		Math.sin(Math.PI / 8),
		"chest rotation x"
	);
	assertNearly(
		Math.abs(chestRotationTrack.values[7]),
		Math.cos(Math.PI / 8),
		"chest rotation w"
	);
}

function testPrefabContract() {
	const loader = new BVHLoader();
	const prefab = loader.parsePrefab(SAMPLE_BVH, { clipName: "prefabMotion" });
	const scene = new Scene();
	const instance = prefab.instantiate(scene);

	assert.ok(prefab.animationBundle);
	assert.equal(prefab.animationBundle.clips[0].name, "prefabMotion");
	assert.ok(instance.root);
	assert.ok(instance.rootEntity !== null);
}

function run() {
	testParseHierarchyAndTracks();
	testPrefabContract();
	console.log("BVH loader tests passed");
}

run();
