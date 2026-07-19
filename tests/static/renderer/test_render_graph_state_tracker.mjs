import assert from "node:assert/strict";

import { RenderGraphCompiler } from "../../../src/rendergraph/RenderGraphCompiler.ts";
import { RenderGraphStateTracker } from "../../../src/rendergraph/RenderGraphStateTracker.ts";

function node(id, resources = [], extra = {}) {
	return {
		id,
		stage: "main-opaque",
		kind: "test",
		label: id,
		resources,
		...extra,
	};
}

{
	const compiler = new RenderGraphCompiler();
	const request = {
		resources: [{ id: "frame:color", origin: "imported" }],
		nodes: [
			node("late", [], { dependsOn: ["first"] }),
			node("independent"),
			node("first"),
		],
	};
	const first = compiler.compile(request);
	const second = compiler.compile(request);
	assert.deepEqual(first.nodes.map((entry) => entry.id), [
		"independent",
		"first",
		"late",
	]);
	assert.deepEqual(second.nodes.map((entry) => entry.id), [
		"independent",
		"first",
		"late",
	]);
	assert.equal(Object.isFrozen(first), true);
	assert.equal(Object.isFrozen(first.liveRanges), true);
	assert.equal(Object.isFrozen(first.nodes[2].dependsOn), true);
	assert.throws(() => first.nodes[2].dependsOn.push("mutate"), TypeError);
}

{
	const compiler = new RenderGraphCompiler();
	const compiled = compiler.compile({
		resources: [
			{ id: "duplicate", origin: "graph" },
			{ id: "duplicate", origin: "graph" },
		],
		nodes: [
			node("duplicate-node"),
			node("duplicate-node"),
			node("missing-dependency", [], { dependsOn: ["absent"] }),
			node("cycle-a", [], { dependsOn: ["cycle-b"] }),
			node("cycle-b", [], { dependsOn: ["cycle-a"] }),
		],
	});
	const codes = compiled.diagnostics.map((diagnostic) => diagnostic.code);
	assert.ok(codes.includes("duplicate-resource"));
	assert.ok(codes.includes("duplicate-node"));
	assert.ok(codes.includes("missing-dependency"));
	assert.equal(codes.filter((code) => code === "cycle").length, 2);
}

{
	const tracker = new RenderGraphStateTracker({ allowImplicitResources: true });
	tracker.beginFrame([{
		id: "frame:color",
		origin: "imported",
		kind: "texture",
		residency: "frame",
		initialContent: "unknown",
	}]);
	const write = tracker.appendStage({
		nodes: [node("write", [{
			resource: "frame:color",
			access: "write",
			usage: "color-attachment",
		}])],
	});
	const read = tracker.appendStage({
		nodes: [node("read", [{
			resource: "frame:color",
			access: "read",
			usage: "sampled",
		}])],
	});
	assert.equal(write.transitions.length, 1);
	assert.equal(read.transitions[0].hazard, "read-after-write");
	assert.equal(read.transitions[0].previousUsage, "color-attachment");
	assert.equal(read.transitions[0].usage, "sampled");
	assert.equal(tracker.getLiveRanges()[0].firstNodeId, "write");
	assert.equal(tracker.getLiveRanges()[0].lastNodeId, "read");
	assert.deepEqual(tracker.getDebugState().current.nodeIds, ["write", "read"]);
}

{
	const tracker = new RenderGraphStateTracker();
	tracker.beginFrame([{
		id: "frame:color",
		origin: "imported",
		initialContent: "valid",
	}]);
	tracker.appendStage({
		nodes: [
			node("read-sampled", [{
				resource: "frame:color",
				access: "read",
				usage: "sampled",
			}]),
			node("read-copy", [{
				resource: "frame:color",
				access: "read",
				usage: "copy-source",
			}]),
			node("write-first", [{
				resource: "frame:color",
				access: "write",
				usage: "copy-target",
			}]),
			node("write-second", [{
				resource: "frame:color",
				access: "write",
				usage: "storage",
			}]),
		],
	});
	assert.deepEqual(
		tracker.getTransitions().map((transition) => transition.reason),
		[undefined, "usage-transition", "write-after-read", "write-after-write"],
	);
	assert.equal(tracker.getShadowDiagnostics().length, 0);
}

{
	const tracker = new RenderGraphStateTracker({ allowImplicitResources: true });
	tracker.beginFrame([]);
	tracker.appendStage({
		nodes: [node("optional", [{
			resource: "missing",
			access: "read",
			usage: "sampled",
			optional: true,
		}])],
	});
	assert.equal(tracker.getTransitions().length, 0);
	assert.equal(tracker.getLiveRanges().length, 0);
	assert.equal(
		tracker.getResourceDebugState().some((entry) => entry.id === "missing"),
		false,
	);
}

{
	const tracker = new RenderGraphStateTracker({ allowImplicitResources: true });
	tracker.beginFrame([]);
	tracker.appendStage({
		nodes: [node("write", [{
			resource: "transient",
			access: "write",
			usage: "storage",
		}])],
	});
	assert.ok(
		tracker.getShadowDiagnostics().some(
			(diagnostic) => diagnostic.code === "implicit-resource-declaration",
		),
	);
	assert.ok(
		tracker.getShadowDiagnostics().some(
			(diagnostic) => diagnostic.code === "implicit-create",
		),
	);
}

{
	const compiler = new RenderGraphCompiler();
	const compiled = compiler.compile({
		resources: [{
			id: "unknown-import",
			origin: "imported",
			initialContent: "unknown",
		}],
		nodes: [node("read-unknown-import", [{
			resource: "unknown-import",
			access: "read",
			usage: "sampled",
		}])],
	});
	assert.equal(compiled.diagnostics.length, 0);
	assert.deepEqual(
		compiled.shadowDiagnostics.map((diagnostic) => diagnostic.code),
		["read-content-unknown"],
	);
}

{
	const tracker = new RenderGraphStateTracker();
	tracker.beginFrame([{
		id: "scratch",
		origin: "graph",
		initialContent: "undefined",
	}]);
	tracker.appendStage({
		nodes: [{
			...node("first"),
			creates: ["scratch"],
			resources: [{
				resource: "scratch",
				access: "write",
				usage: "storage",
			}],
			destroys: ["scratch"],
		}],
	});
	tracker.appendStage({
		nodes: [{
			...node("second"),
			creates: ["scratch"],
			resources: [{
				resource: "scratch",
				access: "write",
				usage: "storage",
			}],
		}],
	});
	assert.deepEqual(
		tracker.getLiveRanges().map((range) => range.generation),
		[1, 2],
	);
	assert.deepEqual(tracker.getLiveRanges()[0], {
		resourceId: "scratch",
		generation: 1,
		firstNodeId: "first",
		lastNodeId: "first",
		createdByNodeId: "first",
		firstUseNodeId: "first",
		lastUseNodeId: "first",
		destroyedByNodeId: "first",
	});
	assert.equal(tracker.getTransitions()[1].previousAccess, undefined);
}

{
	const tracker = new RenderGraphStateTracker();
	tracker.beginFrame([{
		id: "unknown",
		origin: "imported",
		initialContent: "unknown",
	}]);
	tracker.appendStage({
		nodes: [node("read-unknown", [{
			resource: "unknown",
			access: "read",
			usage: "sampled",
		}])],
	});
	assert.ok(
		tracker.getShadowDiagnostics().some(
			(diagnostic) => diagnostic.code === "read-content-unknown",
		),
	);
}

{
	const tracker = new RenderGraphStateTracker();
	tracker.beginFrame([{
		id: "undefined",
		origin: "imported",
		initialContent: "undefined",
	}]);
	tracker.appendStage({
		nodes: [node("read-undefined", [{
			resource: "undefined",
			access: "read",
			usage: "sampled",
		}])],
	});
	assert.equal(tracker.getDiagnostics().length, 0);
	assert.ok(
		tracker.getShadowDiagnostics().some(
			(diagnostic) => diagnostic.code === "read-before-initialize",
		),
	);
	tracker.seal();
	assert.throws(() => tracker.appendStage({ nodes: [] }), /state "sealed"/);
	tracker.abort(new Error("failed"));
	assert.equal(tracker.getDebugState().lastAttempt.state, "aborted");
	assert.equal(tracker.getDebugState().lastSuccessful, null);

	tracker.beginFrame([]);
	tracker.commit();
	assert.equal(tracker.getDebugState().lastSuccessful.state, "committed");
	assert.equal(tracker.getDebugState().lastAttempt.state, "committed");
}

console.log("Render graph state tracker tests passed");
