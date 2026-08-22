import assert from "node:assert/strict";

import { WebGLFrameServices } from "../../../src/backends/webgl/WebGLFrameServices.ts";
import { WebGLFrameSession } from "../../../src/backends/webgl/WebGLFrameSession.ts";

function testSessionClearsActiveFrameState() {
	const session = new WebGLFrameSession();
	const context = {
		attachments: { width: 640.8, height: 359.9 },
	};
	session.presented = true;
	session.begin(context);
	assert.equal(session.context, context);
	assert.equal(session.width, 640);
	assert.equal(session.height, 359);
	assert.equal(session.presented, false);

	session.lightState = { directional: [] };
	session.finish();
	assert.equal(session.context, null);
	assert.equal(session.lightState, null);
}

function testAbortClearsActiveFrameState() {
	const session = new WebGLFrameSession();
	session.begin({ attachments: { width: 1, height: 1 } });
	session.presented = true;
	session.abort();
	assert.equal(session.context, null);
	assert.equal(session.presented, false);
}

function testFrameServicesRequireExplicitPostProcessRuntime() {
	assert.throws(
		() => new WebGLFrameServices({}),
		/explicitly owned post-process runtime/,
	);
}

testSessionClearsActiveFrameState();
testAbortClearsActiveFrameState();
testFrameServicesRequireExplicitPostProcessRuntime();
console.log("WebGL frame session tests passed");
