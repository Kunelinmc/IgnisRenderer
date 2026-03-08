import assert from "node:assert/strict";
import { BlendTree1D } from "../src/animation/BlendTree1D.ts";
import { BlendTreeDirect } from "../src/animation/BlendTreeDirect.ts";
import { AnimationStateMachine } from "../src/animation/AnimationStateMachine.ts";

function nearlyEqual(left, right, epsilon = 1e-6) {
	return Math.abs(left - right) <= epsilon;
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

function run() {
	testBlendTree1D();
	testBlendTreeDirect();
	testStateMachineTransitionsAndTriggerConsumption();
	console.log("Animation state/blend tree tests passed");
}

run();
