import assert from "node:assert/strict";
import {
	computeInteractionOutlineShapeDistance,
	getInteractionOutlineShapeCode,
	resolveInteractionOutlineShape,
} from "../src/interaction/outlineShape.ts";

function run() {
	assert.equal(resolveInteractionOutlineShape(undefined), "circle");
	assert.equal(resolveInteractionOutlineShape("square"), "square");
	assert.equal(resolveInteractionOutlineShape("diamond"), "diamond");
	assert.equal(resolveInteractionOutlineShape("octagon"), "octagon");
	assert.equal(resolveInteractionOutlineShape("invalid"), "circle");

	assert.equal(getInteractionOutlineShapeCode("circle"), 0);
	assert.equal(getInteractionOutlineShapeCode("square"), 1);
	assert.equal(getInteractionOutlineShapeCode("diamond"), 2);
	assert.equal(getInteractionOutlineShapeCode("octagon"), 3);
	assert.equal(getInteractionOutlineShapeCode("invalid"), 0);

	assert.equal(
		computeInteractionOutlineShapeDistance(3, 4, "circle"),
		5
	);
	assert.equal(
		computeInteractionOutlineShapeDistance(1, 1, "diamond"),
		2
	);
	assert.equal(
		computeInteractionOutlineShapeDistance(1, 0, "square"),
		Math.SQRT2
	);
	assert.ok(
		computeInteractionOutlineShapeDistance(1, 1, "octagon") >
			computeInteractionOutlineShapeDistance(0.7, 0.7, "octagon")
	);

	console.log("Interaction outline shape tests passed");
}

run();
