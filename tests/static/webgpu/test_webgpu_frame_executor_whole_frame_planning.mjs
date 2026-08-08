import assert from "node:assert/strict";
import { EMPTY_SHADOW_FRAME_PLAN } from "../../../src/lights/shadows/ShadowFramePlan.ts";

assert.ok(Object.isFrozen(EMPTY_SHADOW_FRAME_PLAN));
assert.equal(EMPTY_SHADOW_FRAME_PLAN.lights.length, 0);
console.log("WebGPU whole-frame planning accepts an immutable empty shadow plan");
