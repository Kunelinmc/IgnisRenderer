import assert from "node:assert/strict";
import { BlendTree1D } from "../src/animation/BlendTree1D.ts";
import { BlendTree2D } from "../src/animation/BlendTree2D.ts";
import { BlendTreeDirect } from "../src/animation/BlendTreeDirect.ts";
import { AnimationClip } from "../src/animation/AnimationClip.ts";
import { AnimationSystem } from "../src/animation/AnimationSystem.ts";
import { AnimationStateMachine } from "../src/animation/AnimationStateMachine.ts";
import { Node } from "../src/core/Node.ts";
import { AnimationRuntime } from "../src/simulation/animation/AnimationRuntime.ts";

function nearlyEqual(left, right, epsilon = 1e-6) {
	return Math.abs(left - right) <= epsilon;
}

function sumWeights(weights) {
	return weights.reduce((sum, entry) => sum + entry.weight, 0);
}

function getWeight(weights, clipName) {
	const found = weights.find((entry) => entry.clipName === clipName);
	return found ? found.weight : 0;
}

function testBlendTree1D() {
	const tree = new BlendTree1D({
		name: "locomotion",
		parameter: "speed",
		children: [
			{ clipName: "idle", threshold: 0 },
			{ clipName: "walk", threshold: 1 },
			{ clipName: "run", threshold: 3 },
		],
	});

	const betweenIdleAndWalk = tree.evaluate(0.25);
	assert.equal(betweenIdleAndWalk.length, 2);
	assert.equal(betweenIdleAndWalk[0].clipName, "idle");
	assert.equal(betweenIdleAndWalk[1].clipName, "walk");
	assert.ok(nearlyEqual(betweenIdleAndWalk[0].weight, 0.75));
	assert.ok(nearlyEqual(betweenIdleAndWalk[1].weight, 0.25));

	const maxed = tree.evaluate(10);
	assert.deepEqual(maxed, [{ clipName: "run", weight: 1 }]);
}

function testBlendTreeDirect() {
	const tree = new BlendTreeDirect({
		name: "upper-body",
		children: [
			{ clipName: "aim", parameter: "aim" },
			{ clipName: "reload", parameter: "reload" },
		],
	});

	const weights = tree.evaluate(
		new Map([
			["aim", 3],
			["reload", 1],
		])
	);
	assert.equal(weights.length, 2);
	assert.ok(nearlyEqual(weights[0].weight, 0.75));
	assert.ok(nearlyEqual(weights[1].weight, 0.25));

	const fallback = tree.evaluate(new Map());
	assert.equal(fallback[0].weight, 1);
	assert.equal(fallback[1].weight, 0);
}

function testBlendTree2DDirectional() {
	const tree = new BlendTree2D({
		name: "locomotion-2d-directional",
		parameterX: "moveX",
		parameterY: "moveY",
		blendMode: "directional",
		children: [
			{ clipName: "idle", positionX: 0, positionY: 0 },
			{ clipName: "walkForward", positionX: 0, positionY: 1 },
			{ clipName: "runForward", positionX: 0, positionY: 2 },
			{ clipName: "walkRight", positionX: 1, positionY: 0 },
			{ clipName: "walkLeft", positionX: -1, positionY: 0 },
		],
	});

	const forwardRun = tree.evaluate(0, 1.8);
	assert.ok(nearlyEqual(sumWeights(forwardRun), 1));
	assert.ok(getWeight(forwardRun, "runForward") > getWeight(forwardRun, "walkForward"));
	assert.ok(getWeight(forwardRun, "runForward") > 0.5);

	const diagonal = tree.evaluate(0.7, 0.7);
	assert.ok(nearlyEqual(sumWeights(diagonal), 1));
	assert.ok(getWeight(diagonal, "walkForward") > 0.2);
	assert.ok(getWeight(diagonal, "walkRight") > 0.2);
	assert.ok(getWeight(diagonal, "walkLeft") < 0.05);
}

function testBlendTree2DTriangulation() {
	const tree = new BlendTree2D({
		name: "locomotion-2d-triangulation",
		parameterX: "moveX",
		parameterY: "moveY",
		blendMode: "triangulation",
		children: [
			{ clipName: "idle", positionX: 0, positionY: 0 },
			{ clipName: "forward", positionX: 0, positionY: 1 },
			{ clipName: "right", positionX: 1, positionY: 0 },
		],
	});

	const inside = tree.evaluate(0.3, 0.3);
	assert.ok(nearlyEqual(sumWeights(inside), 1));
	assert.ok(nearlyEqual(getWeight(inside, "idle"), 0.4));
	assert.ok(nearlyEqual(getWeight(inside, "forward"), 0.3));
	assert.ok(nearlyEqual(getWeight(inside, "right"), 0.3));

	const outside = tree.evaluate(1.2, 0.1);
	assert.ok(nearlyEqual(sumWeights(outside), 1));
	assert.ok(getWeight(outside, "right") > 0.85);
	assert.ok(getWeight(outside, "forward") < 0.15);
}

function createEmptyClip(name) {
	return new AnimationClip({
		name,
		duration: 1,
		tracks: [],
	});
}

function testBlendTree2DStateRuntimeIntegration() {
	const system = new AnimationSystem();
	const mixer = system.createMixer(new Node({ name: "root" }));
	mixer.addClip(createEmptyClip("idle"));
	mixer.addClip(createEmptyClip("forward"));
	mixer.addClip(createEmptyClip("right"));

	mixer.addBlendTree2D(
		new BlendTree2D({
			name: "locomotion-2d-runtime",
			parameterX: "moveX",
			parameterY: "moveY",
			blendMode: "triangulation",
			children: [
				{ clipName: "idle", positionX: 0, positionY: 0 },
				{ clipName: "forward", positionX: 0, positionY: 1 },
				{ clipName: "right", positionX: 1, positionY: 0 },
			],
		})
	);

	const machine = new AnimationStateMachine({
		name: "locomotion-runtime",
		parameters: [
			{ name: "moveX", type: "float", defaultValue: 0 },
			{ name: "moveY", type: "float", defaultValue: 0 },
		],
		states: [
			{
				name: "locomotion",
				motion: { type: "blendtree2d", treeName: "locomotion-2d-runtime" },
			},
		],
		initialState: "locomotion",
	});

	mixer.addStateMachine("Base Layer", machine);
	machine.setParameter("moveX", 0.3);
	machine.setParameter("moveY", 0.3);

	const runtime = new AnimationRuntime();
	runtime.update(system, 0.016, new Map());

	const layer = mixer.getLayer("Base Layer");
	assert.ok(layer);
	assert.ok(nearlyEqual(layer.getAction("idle")?.weight ?? 0, 0.4));
	assert.ok(nearlyEqual(layer.getAction("forward")?.weight ?? 0, 0.3));
	assert.ok(nearlyEqual(layer.getAction("right")?.weight ?? 0, 0.3));
}

function testStateMachineTransitionsAndTriggerConsumption() {
	const machine = new AnimationStateMachine({
		name: "combat",
		parameters: [
			{ name: "speed", type: "float", defaultValue: 0 },
			{ name: "fire", type: "trigger", defaultValue: false },
		],
		states: [
			{ name: "idle", motion: { type: "clip", clipName: "idle" } },
			{ name: "run", motion: { type: "clip", clipName: "run" } },
			{ name: "shoot", motion: { type: "clip", clipName: "shoot" } },
		],
		transitions: [
			{
				from: "idle",
				to: "run",
				duration: 0.2,
				conditions: [{ parameter: "speed", operator: ">", value: 0.5 }],
			},
			{
				from: "run",
				to: "shoot",
				duration: 0.1,
				conditions: [{ parameter: "fire", operator: "trigger" }],
			},
			{
				from: "shoot",
				to: "idle",
				duration: 0.1,
				hasExitTime: true,
				exitTime: 0.6,
			},
		],
		initialState: "idle",
	});

	assert.equal(machine.currentStateName, "idle");

	machine.setParameter("speed", 1);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "run");
	machine.update(0, 0.2);
	assert.equal(machine.currentStateName, "run");

	machine.setParameter("fire", true);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "shoot");
	assert.equal(machine.getParameter("fire"), false);
	machine.update(0, 0.1);
	assert.equal(machine.currentStateName, "shoot");

	machine.update(0.5, 0.016);
	assert.equal(machine.currentStateName, "shoot");
	assert.equal(machine.transitionState, null);

	machine.update(0.75, 0.016);
	assert.equal(machine.transitionState?.to, "idle");
	machine.update(0, 0.1);
	assert.equal(machine.currentStateName, "idle");
}

function testTriggerPersistsUntilTransitionConsumesIt() {
	const machine = new AnimationStateMachine({
		name: "trigger-persistence",
		parameters: [{ name: "fire", type: "trigger", defaultValue: false }],
		states: [
			{ name: "idle", motion: { type: "clip", clipName: "idle" } },
			{ name: "shoot", motion: { type: "clip", clipName: "shoot" } },
		],
		transitions: [
			{
				from: "idle",
				to: "shoot",
				duration: 0.1,
				hasExitTime: true,
				exitTime: 0.75,
				conditions: [{ parameter: "fire", operator: "trigger" }],
			},
		],
		initialState: "idle",
	});

	machine.setParameter("fire", true);
	machine.update(0.2, 0.016);
	assert.equal(machine.transitionState, null);
	assert.equal(machine.getParameter("fire"), true);

	machine.update(0.9, 0.016);
	assert.equal(machine.transitionState?.to, "shoot");
	assert.equal(machine.getParameter("fire"), false);
}

function testStateMachinePriorityAnyStateAndInterrupt() {
	const machine = new AnimationStateMachine({
		name: "priority-any-interrupt",
		parameters: [
			{ name: "speed", type: "float", defaultValue: 0 },
			{ name: "jump", type: "trigger", defaultValue: false },
			{ name: "die", type: "trigger", defaultValue: false },
		],
		states: [
			{ name: "idle", motion: { type: "clip", clipName: "idle" } },
			{ name: "walk", motion: { type: "clip", clipName: "walk" } },
			{ name: "run", motion: { type: "clip", clipName: "run" } },
			{ name: "jump", motion: { type: "clip", clipName: "jump" } },
			{ name: "dead", motion: { type: "clip", clipName: "dead" } },
		],
		transitions: [
			{
				from: "idle",
				to: "walk",
				duration: 0.3,
				priority: 1,
				conditions: [{ parameter: "speed", operator: ">", value: 0.5 }],
			},
			{
				from: "idle",
				to: "run",
				duration: 0.8,
				priority: 10,
				interruptible: true,
				conditions: [{ parameter: "speed", operator: ">", value: 0.5 }],
			},
			{
				from: "run",
				to: "jump",
				duration: 0.2,
				priority: 20,
				canInterrupt: true,
				conditions: [{ parameter: "jump", operator: "trigger" }],
			},
		],
		anyStateTransitions: [
			{
				to: "dead",
				duration: 0.05,
				priority: 100,
				canInterrupt: true,
				conditions: [{ parameter: "die", operator: "trigger" }],
			},
		],
		initialState: "idle",
	});

	machine.setParameter("speed", 1);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "run");

	machine.setParameter("jump", true);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.from, "run");
	assert.equal(machine.transitionState?.to, "jump");

	machine.setParameter("die", true);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "dead");
}

function testSubStateMachineTransitions() {
	const machine = new AnimationStateMachine({
		name: "sub-state-machine",
		parameters: [
			{ name: "speed", type: "float", defaultValue: 0 },
			{ name: "attack", type: "trigger", defaultValue: false },
			{ name: "returnToLocomotion", type: "trigger", defaultValue: false },
		],
		states: [{ name: "combat", motion: { type: "clip", clipName: "combat" } }],
		subStateMachines: [
			{
				name: "locomotion",
				initialState: "idle",
				states: [
					{
						name: "idle",
						motion: { type: "clip", clipName: "idle" },
					},
					{
						name: "run",
						motion: { type: "clip", clipName: "run" },
					},
				],
				transitions: [
					{
						from: "idle",
						to: "run",
						duration: 0.1,
						conditions: [{ parameter: "speed", operator: ">", value: 0.5 }],
					},
					{
						from: "run",
						to: "idle",
						duration: 0.1,
						conditions: [{ parameter: "speed", operator: "<=", value: 0.1 }],
					},
				],
			},
		],
		transitions: [
			{
				from: "locomotion",
				to: "combat",
				duration: 0.05,
				conditions: [{ parameter: "attack", operator: "trigger" }],
			},
			{
				from: "combat",
				to: "locomotion",
				duration: 0.05,
				conditions: [{ parameter: "returnToLocomotion", operator: "trigger" }],
			},
		],
		initialState: "locomotion",
	});

	assert.equal(machine.currentStateName, "locomotion/idle");
	assert.equal(
		machine.getStateDefinition("locomotion")?.name,
		"locomotion/idle"
	);

	machine.setParameter("speed", 1);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "locomotion/run");
	machine.update(0, 0.1);
	assert.equal(machine.currentStateName, "locomotion/run");

	machine.setParameter("attack", true);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.from, "locomotion/run");
	assert.equal(machine.transitionState?.to, "combat");
	machine.update(0, 0.05);
	assert.equal(machine.currentStateName, "combat");

	machine.setParameter("returnToLocomotion", true);
	machine.update(0, 0.016);
	assert.equal(machine.transitionState?.to, "locomotion/idle");
	machine.update(0, 0.05);
	assert.equal(machine.currentStateName, "locomotion/idle");
}

function run() {
	testBlendTree1D();
	testBlendTreeDirect();
	testBlendTree2DDirectional();
	testBlendTree2DTriangulation();
	testBlendTree2DStateRuntimeIntegration();
	testStateMachineTransitionsAndTriggerConsumption();
	testTriggerPersistsUntilTransitionConsumesIt();
	testStateMachinePriorityAnyStateAndInterrupt();
	testSubStateMachineTransitions();
	console.log("Animation state/blend tree tests passed");
}

run();
