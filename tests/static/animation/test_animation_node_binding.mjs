import assert from "node:assert/strict";
import { AnimationClip } from "../../../src/animation/AnimationClip.ts";
import { AnimationSystem } from "../../../src/animation/AnimationSystem.ts";
import { KeyframeTrack } from "../../../src/animation/KeyframeTrack.ts";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { ECSWorld } from "../../../src/ecs/ECSWorld.ts";
import { AnimationRuntime } from "../../../src/simulation/animation/AnimationRuntime.ts";

function run() {
	const scene = new Scene();
	const root = scene.root;
	const arm = new Node({ name: "arm" });
	scene.add(arm);
	scene.updateWorldMatrices();
	const world = new ECSWorld(scene);
	const armEntity = world.getEntityByNode(arm);
	assert.ok(armEntity !== null);

	const system = new AnimationSystem();
	const mixer = system.createMixer(root);
	mixer.bindNode("/node/arm", arm);
	mixer.addClip(
		new AnimationClip({
			name: "move-arm",
			duration: 1,
			tracks: [
				new KeyframeTrack({
					binding: {
						targetType: "node",
						targetPath: "/node/arm",
						property: "translation",
					},
					times: [0, 1],
					values: [0, 0, 0, 8, 0, 0],
					valueSize: 3,
				}),
			],
		}),
	);
	mixer.clipAction("move-arm").play();

	const runtime = new AnimationRuntime();
	runtime.update(system, 0.5, new Map(), scene);
	assert.ok(arm.position.x > 3.9);
	scene.updateWorldMatrices();
	assert.ok(world.getComponent(armEntity, "LocalTransform").positionX > 3.9);
	world.destroy();

	console.log("Animation node binding tests passed");
}

run();
