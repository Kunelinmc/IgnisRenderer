import assert from "node:assert/strict";

import { createBrowserTestSession } from "../../browser/helpers/browserTestSession.ts";
import { withBrowserFixture } from "../../browser/helpers/browserFixture.ts";

// Browser transport is replaced here; session acquisition and cleanup are real.
function createPageBridge() {
	let handles = 0;
	return {
		get handles() { return handles; },
		async goto() {},
		async evaluateHandle(setup) {
			const session = await setup();
			handles++;
			return {
				async evaluate(callback) {
					return structuredClone(await callback(session));
				},
				async dispose() { handles--; },
			};
		},
	};
}

{
	const events = [];
	const session = await createBrowserTestSession(async (defer) => {
		defer(() => { events.push("canvas"); });
		defer(async () => {
			await Promise.resolve();
			events.push("renderer");
		});
		return { pixel: [20, 120, 220] };
	});
	assert.deepEqual(session.result, { pixel: [20, 120, 220] });
	assert.deepEqual(events, [], "Resources must remain alive for assertions and screenshots.");
	await Promise.all([session.dispose(), session.dispose()]);
	await session.dispose();
	assert.deepEqual(events, ["renderer", "canvas"], "Cleanup must await reverse-order disposal once.");
}

{
	let live = false;
	const failure = new Error("Initialization failed");
	await assert.rejects(createBrowserTestSession(async (defer) => {
		live = true;
		defer(() => { live = false; });
		throw failure;
	}), (error) => error === failure);
	assert.equal(live, false, "Failed setup must release resources acquired before the error.");
}

{
	let canvasRemoved = false;
	const failure = new Error("Renderer disposal failed");
	const session = await createBrowserTestSession(async (defer) => {
		defer(() => { canvasRemoved = true; });
		defer(() => { throw failure; });
		return null;
	});
	await assert.rejects(session.dispose(), (error) => error === failure);
	assert.equal(canvasRemoved, true, "A failed cleanup must not prevent remaining cleanup.");
}

{
	const setupFailure = new Error("Setup failed");
	const cleanupFailure = new Error("Cleanup failed");
	await assert.rejects(createBrowserTestSession(async (defer) => {
		defer(() => { throw cleanupFailure; });
		throw setupFailure;
	}), (error) => {
		assert.ok(error instanceof AggregateError);
		assert.deepEqual(error.errors, [setupFailure, cleanupFailure]);
		return true;
	});
}

{
	const page = createPageBridge();
	let live = false;
	const value = await withBrowserFixture(page, () => createBrowserTestSession(async (defer) => {
		live = true;
		defer(() => { live = false; });
		return { ready: true };
	}), async (result) => {
		assert.equal(live, true);
		assert.deepEqual(result, { ready: true });
		await Promise.resolve();
		assert.equal(live, true, "Async screenshot work must finish before disposal.");
		return 42;
	});
	assert.equal(value, 42);
	assert.equal(live, false);
	assert.equal(page.handles, 0);
}

{
	const page = createPageBridge();
	let live = false;
	const failure = new Error("Assertion failed");
	await assert.rejects(withBrowserFixture(page, () => createBrowserTestSession(async (defer) => {
		live = true;
		defer(() => { live = false; });
		return null;
	}), () => { throw failure; }), (error) => error === failure);
	assert.equal(live, false, "A failed assertion must still dispose the browser session.");
	assert.equal(page.handles, 0);
}

{
	const page = createPageBridge();
	const assertionFailure = new Error("Assertion failed");
	const cleanupFailure = new Error("GPU disposal failed");
	await assert.rejects(withBrowserFixture(page, () => createBrowserTestSession(async (defer) => {
		defer(() => { throw cleanupFailure; });
		return null;
	}), () => { throw assertionFailure; }), (error) => {
		assert.ok(error instanceof AggregateError);
		assert.deepEqual(error.errors, [assertionFailure, cleanupFailure]);
		return true;
	});
	assert.equal(page.handles, 0, "A failed resource cleanup must still release the handle.");
}

console.log("Browser fixture lifecycle tests passed.");
