import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Node } from "../src/core/Node.ts";
import { AnimationSystem } from "../src/animation/AnimationSystem.ts";
import { AnimationClip } from "../src/animation/AnimationClip.ts";
import { KeyframeTrack } from "../src/animation/KeyframeTrack.ts";
import { AnimationRuntime } from "../src/simulation/animation/AnimationRuntime.ts";

function run() {
	const scene = new Scene();
	const root = scene.root;
	const arm = new Node({ name: "arm" });
	scene.add(arm);
	scene.updateWorldMatrices();

	const entityId = arm.entityId;
	assert.ok(entityId !== null);

	const system = new AnimationSystem();
	const mixer = system.createMixer(root);
	mixer.bindEntity("/entity/arm", entityId);
	mixer.addClip(
		new AnimationClip({
			name: "move-arm",
			duration: 1,
			tracks: [
				new KeyframeTrack({
					binding: {
						targetType: "node",
						targetPath: "/entity/arm",
						property: "translation",
					},
					times: [0, 1],
					values: [0, 0, 0, 8, 0, 0],
					valueSize: 3,
				}),
			],
		})
	);
	mixer.clipAction("move-arm").play();

	const runtime = new AnimationRuntime();
	runtime.update(system, 0.5, new Map(), scene);
	const local = scene.ecs.getComponent(entityId, "LocalTransform");
	assert.ok(local.positionX > 3.9);
	assert.ok(arm.position.x > 3.9);

	console.log("Animation entity binding tests passed");
}

run();
