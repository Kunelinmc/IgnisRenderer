import assert from "node:assert/strict";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { ECSWorld } from "../../../src/ecs/ECSWorld.ts";

function run() {
	const scene = new Scene();
	const parent = new Node({ name: "parent" });
	const child = new Node({ name: "child" });
	parent.addChild(child);
	scene.add(parent);
	scene.updateWorldMatrices();

	assert.equal("ecs" in scene, false);
	assert.equal("entityId" in child, false);

	const world = new ECSWorld(scene);
	const secondWorld = new ECSWorld(scene);
	const parentEntity = world.getEntityByNode(parent);
	const childEntity = world.getEntityByNode(child);
	assert.ok(parentEntity !== null);
	assert.ok(childEntity !== null);
	assert.ok(secondWorld.getEntityByNode(child) !== null);
	assert.equal(world.getNodeByEntity(childEntity), child);

	const local = world.getComponent(childEntity, "LocalTransform");
	const worldTransform = world.getComponent(childEntity, "WorldTransform");
	assert.ok(local);
	assert.ok(worldTransform);
	local.positionX = 12;
	world.setComponent(childEntity, "LocalTransform", local);
	assert.equal(child.position.x, 12);

	scene.updateWorldMatrices();
	assert.equal(world.getComponent(childEntity, "LocalTransform"), local);
	assert.equal(world.getComponent(childEntity, "WorldTransform"), worldTransform);
	const secondChildEntity = secondWorld.getEntityByNode(child);
	assert.equal(
		secondWorld.getComponent(secondChildEntity, "LocalTransform").positionX,
		12,
	);
	const versionAfterChange = world.version;
	scene.updateWorldMatrices();
	assert.equal(world.version, versionAfterChange);

	child.visible = false;
	child.name = "renamed";
	assert.equal(world.getComponent(childEntity, "Visibility").visible, false);
	assert.equal(world.getComponent(childEntity, "Name").value, "renamed");
	assert.match(world.getComponent(childEntity, "PathBinding").path, /renamed/);

	scene.add(child);
	assert.equal(world.getEntityByNode(child), childEntity);
	assert.equal(
		world.getComponent(childEntity, "Hierarchy").parent,
		world.getEntityByNode(scene.root),
	);
	assert.deepEqual(Array.from(world.getComponent(parentEntity, "Hierarchy").children), []);

	assert.throws(
		() => world.setComponent(childEntity, "WorldTransform", worldTransform),
		/read-only/,
	);
	assert.throws(() => world.removeComponent(childEntity, "Name"), /required/);
	assert.throws(() => world.destroyEntity(childEntity), /scene-backed/);

	world.destroy();
	scene.remove(child);
	assert.equal(world.getEntityByNode(child), null);
	assert.equal(secondWorld.getEntityByNode(child), null);
	secondWorld.destroy();

}

function testDerivedViewsRejectMutationAndStayLive() {
	const scene = new Scene();
	const parent = new Node({ name: "parent" });
	const child = parent.addChild(new Node({ name: "child" }));
	scene.add(parent);
	const world = new ECSWorld(scene);
	const entity = world.getEntityByNode(child);
	const parentEntity = world.getEntityByNode(parent);
	const hierarchy = world.getComponent(entity, "Hierarchy");
	const children = world.getComponent(parentEntity, "Hierarchy").children;
	const transform = world.getComponent(entity, "WorldTransform");
	const matrix = transform.matrix;
	const rows = matrix.elements;
	const row = rows[0];
	const nodeRef = world.getComponent(entity, "NodeRef");
	const version = world.version;
	for (const mutate of [
		() => { hierarchy.parent = null; },
		() => { children.push(123); },
		() => { transform.matrix = parent.worldMatrix; },
		() => { rows[0] = [9, 9, 9, 9]; },
		() => { row[3] = 99; },
		() => { matrix.multiply(parent.worldMatrix); },
		() => { nodeRef.node = parent; },
		() => { world.getComponent(entity, "PathBinding").path = "bad"; },
		() => { world.getComponent(entity, "NodeKind").kind = "camera"; },
		() => { Object.getOwnPropertyDescriptor(matrix, "elements").value[0][3] = 99; },
		() => { Object.defineProperty(hierarchy, "parent", { value: null }); },
		() => { delete hierarchy.parent; },
		() => { Object.setPrototypeOf(hierarchy, null); },
		() => { Object.preventExtensions(hierarchy); },
	]) assert.throws(mutate, /read-only/);
	assert.equal(world.version, version);
	assert.equal(nodeRef.node, child);
	assert.deepEqual(Array.from(children), [entity]);

	child.position.x = 4;
	scene.updateWorldMatrices();
	assert.equal(world.getComponent(entity, "WorldTransform"), transform);
	assert.equal(transform.matrix, matrix);
	assert.equal(matrix.elements, rows);
	assert.equal(rows[0], row);
	assert.equal(row[3], 4);
	assert.equal(matrix.clone().elements[0][3], 4);
	scene.add(child);
	assert.deepEqual(Array.from(children), []);
	assert.equal(hierarchy.parent, world.getEntityByNode(scene.root));
	nodeRef.node.name = "authored";
	assert.equal(world.getComponent(entity, "Name").value, "authored");

	const standalone = world.createEntity();
	world.setComponent(standalone, "Hierarchy", { parent: null, children: [] });
	world.getComponent(standalone, "Hierarchy").children.push(entity);
	assert.deepEqual(world.getComponent(standalone, "Hierarchy").children, [entity]);
	world.destroy();
}

function testReentrantMetadataPreservesFinalNodeState() {
	const scene = new Scene();
	const first = new Node({ name: "first" });
	const second = new Node({ name: "second" });
	scene.add(first);
	scene.add(second);
	const worlds = [new ECSWorld(scene), new ECSWorld(scene)];
	scene.addNodeLifecycleListener({
		nodeStateChanged: ({ node, field }) => {
			if (node === first && field === "name" && first.name === "requested") {
				second.name = "nested";
				worlds[0].setComponent(worlds[0].getEntityByNode(second), "Visibility", {
					visible: false,
				});
				first.name = "normalized";
			}
		},
	});
	worlds[0].setComponent(worlds[0].getEntityByNode(first), "Name", { value: "requested" });
	for (const world of worlds) {
		const firstEntity = world.getEntityByNode(first);
		const secondEntity = world.getEntityByNode(second);
		assert.equal(world.getComponent(firstEntity, "Name").value, "normalized");
		assert.equal(world.getComponent(secondEntity, "Name").value, "nested");
		assert.equal(world.getComponent(secondEntity, "Visibility").visible, false);
		assert.match(world.getComponent(firstEntity, "PathBinding").path, /normalized_/);
		assert.match(world.getComponent(secondEntity, "PathBinding").path, /nested_/);
		world.destroy();
	}
}

function testMetadataListenerCanDetachOrDestroyDuringWrite() {
	for (const action of ["detach", "destroy"]) {
		const scene = new Scene();
		const node = new Node({ name: "before" });
		scene.add(node);
		const world = new ECSWorld(scene);
		const entity = world.getEntityByNode(node);
		let finalVersion;
		scene.addNodeLifecycleListener({
			nodeStateChanged: () => {
				if (action === "detach") scene.remove(node);
				else world.destroy();
				finalVersion = world.version;
			},
		});
		world.setComponent(entity, "Name", { value: "after" });
		assert.equal(world.hasEntity(entity), false);
		assert.equal(world.getComponent(entity, "Name"), undefined);
		assert.equal(world.version, finalVersion);
		world.destroy();
	}
}

function testAliasedMetadataWritesAdvanceVersion() {
	const scene = new Scene();
	const node = new Node({ name: "before" });
	scene.add(node);
	const world = new ECSWorld(scene);
	const entity = world.getEntityByNode(node);
	for (const [name, field, value] of [["Name", "value", "after"], ["Visibility", "visible", false]]) {
		const component = world.getComponent(entity, name);
		component[field] = value;
		const version = world.version;
		world.setComponent(entity, name, component);
		assert.ok(world.version > version);
	}
	assert.equal(node.name, "after");
	assert.equal(node.visible, false);
	world.destroy();
}

function testEqualMatricesStillPublishAuthoredTransformChanges() {
	for (const zeroScale of [false, true]) {
		const scene = new Scene();
		const node = new Node();
		if (zeroScale) node.scale.set(0, 0, 0);
		scene.add(node);
		const world = new ECSWorld(scene);
		const entity = world.getEntityByNode(node);
		const revision = node.worldTransformRevision;
		if (zeroScale) node.quaternion.set(0, 0, 1, 0);
		else node.quaternion.set(0, 0, 0, -1);
		scene.updateWorldMatrices();
		const local = world.getComponent(entity, "LocalTransform");
		assert.equal(local.rotationZ, zeroScale ? 1 : 0);
		assert.equal(local.rotationW, zeroScale ? 0 : -1);
		assert.equal(node.worldTransformRevision, revision);
		const version = world.version;
		scene.updateWorldMatrices();
		assert.equal(world.version, version);
		world.destroy();
	}
}

function testDirectMatrixResolutionRetainsPendingProjection() {
	const scene = new Scene();
	const parent = new Node();
	const child = parent.addChild(new Node());
	scene.add(parent);
	const worlds = [new ECSWorld(scene), new ECSWorld(scene)];
	parent.position.x = 7;
	child.position.y = 3;
	parent.updateWorldMatrix(scene.root.worldMatrix, []);
	parent.updateWorldMatrix(scene.root.worldMatrix);
	scene.updateWorldMatrices();
	for (const world of worlds) {
		const entity = world.getEntityByNode(child);
		assert.equal(world.getComponent(entity, "LocalTransform").positionY, 3);
		assert.equal(world.getComponent(entity, "WorldTransform").matrix.elements[0][3], 7);
		assert.equal(world.getComponent(entity, "WorldTransform").matrix.elements[1][3], 3);
		const version = world.version;
		scene.updateWorldMatrices();
		assert.equal(world.version, version);
		world.destroy();
	}
}

function testReentrantTransformPublicationPreservesOtherListeners() {
	const scene = new Scene();
	const first = new Node();
	const second = new Node();
	scene.add(first);
	scene.add(second);
	let nested = false;
	scene.addNodeLifecycleListener({
		transformsChanged: () => {
			if (nested) return;
			nested = true;
			second.position.x = 9;
			scene.updateWorldMatrices();
		},
	});
	const world = new ECSWorld(scene);
	nested = false;
	second.position.x = 0;
	scene.updateWorldMatrices();
	nested = false;
	first.position.x = 4;
	scene.updateWorldMatrices();
	assert.equal(world.getComponent(world.getEntityByNode(first), "LocalTransform").positionX, 4);
	assert.equal(world.getComponent(world.getEntityByNode(second), "LocalTransform").positionX, 9);
	world.destroy();
}

function testDetachedTransformChangesDoNotPublishEmptyBatches() {
	const scene = new Scene();
	const node = new Node();
	scene.add(node);
	const world = new ECSWorld(scene);
	const batches = [];
	scene.addNodeLifecycleListener({ transformsChanged: (nodes) => batches.push([...nodes]) });
	node.setPosition(2, 0, 0);
	scene.remove(node);
	scene.updateWorldMatrices();
	assert.deepEqual(batches, []);
	assert.equal(world.getEntityByNode(node), null);
	world.destroy();
}

function testListenerErrorsDoNotSkipOtherWorlds() {
	const scene = new Scene();
	const node = new Node({ name: "before" });
	let fail = false;
	const failure = new Error("listener failure");
	const throwWhenEnabled = () => { if (fail) throw failure; };
	scene.addNodeLifecycleListener({
		nodeAttached: throwWhenEnabled,
		nodeDetached: throwWhenEnabled,
		nodeStateChanged: throwWhenEnabled,
		transformsChanged: throwWhenEnabled,
	});
	const worlds = [new ECSWorld(scene), new ECSWorld(scene)];
	fail = true;
	assert.throws(() => scene.add(node), (error) => error === failure);
	assert.throws(() => worlds[0].setComponent(worlds[0].getEntityByNode(node), "Name", {
		value: "after",
	}), (error) => error === failure);
	node.position.x = 5;
	assert.throws(() => scene.updateWorldMatrices(), (error) => error === failure);
	for (const world of worlds) {
		const entity = world.getEntityByNode(node);
		assert.ok(entity !== null);
		assert.equal(world.getComponent(entity, "Name").value, "after");
		assert.equal(world.getComponent(entity, "WorldTransform").matrix.elements[0][3], 5);
	}
	assert.throws(() => scene.remove(node), (error) => error === failure);
	for (const world of worlds) {
		assert.equal(world.getEntityByNode(node), null);
		world.destroy();
	}
}

function testReentrantAttachmentUsesCurrentOwnership() {
	const scene = new Scene();
	const removed = new Node();
	const parent = new Node();
	const child = new Node();
	const existing = new Node();
	scene.add(existing);
	scene.addNodeLifecycleListener({
		nodeAttached: ({ child: attached }) => {
			if (attached === removed) scene.remove(removed);
			if (attached === parent) {
				parent.addChild(child);
				parent.addChild(existing);
			}
		},
	});
	const world = new ECSWorld(scene);
	const existingEntity = world.getEntityByNode(existing);
	scene.add(removed);
	assert.equal(world.getEntityByNode(removed), null);
	scene.add(parent);
	const parentEntity = world.getEntityByNode(parent);
	const childEntity = world.getEntityByNode(child);
	assert.equal(world.getEntityByNode(existing), existingEntity);
	assert.equal(world.query(["NodeRef"]).length, 4);
	assert.deepEqual(Array.from(world.getComponent(parentEntity, "Hierarchy").children), [
		childEntity, existingEntity,
	]);
	assert.equal(world.getComponent(childEntity, "Hierarchy").parent, parentEntity);
	assert.equal(world.getComponent(existingEntity, "Hierarchy").parent, parentEntity);
	assert.deepEqual(Array.from(world.getComponent(world.getEntityByNode(scene.root),
		"Hierarchy").children), [parentEntity]);
	world.destroy();
}

function testReentrantDetachmentRemovesOriginalProjectedDescendants() {
	const scene = new Scene();
	const parent = new Node();
	const removedChild = parent.addChild(new Node());
	const retainedChild = parent.addChild(new Node());
	scene.add(parent);
	scene.addNodeLifecycleListener({
		nodeDetached: ({ child }) => {
			if (child !== parent) return;
			parent.removeChild(removedChild);
			scene.add(retainedChild);
		},
	});
	const world = new ECSWorld(scene);
	const retainedEntity = world.getEntityByNode(retainedChild);
	scene.remove(parent);
	assert.equal(world.getEntityByNode(parent), null);
	assert.equal(world.getEntityByNode(removedChild), null);
	assert.equal(world.getEntityByNode(retainedChild), retainedEntity);
	assert.equal(world.query(["NodeRef"]).length, 2);
	assert.equal(world.getComponent(retainedEntity, "Hierarchy").parent,
		world.getEntityByNode(scene.root));
	world.destroy();
}

function testReentrantReattachmentPreservesProjection() {
	const scene = new Scene();
	const node = new Node();
	scene.add(node);
	let reattach = true;
	scene.addNodeLifecycleListener({
		nodeDetached: ({ child }) => {
			if (child === node && reattach) {
				reattach = false;
				scene.add(node);
			}
		},
	});
	const world = new ECSWorld(scene);
	const entity = world.getEntityByNode(node);
	scene.remove(node);
	assert.equal(world.getEntityByNode(node), entity);
	assert.equal(world.hasEntity(entity), true);
	assert.equal(world.getComponent(entity, "Hierarchy").parent,
		world.getEntityByNode(scene.root));
	world.destroy();
}

function testReattachedSubtreesReconcileRemovedAndAddedDescendants() {
	const scene = new Scene();
	const parent = new Node();
	const removed = parent.addChild(new Node());
	const retained = parent.addChild(new Node({ name: "before" }));
	const added = new Node();
	scene.add(parent);
	scene.addNodeLifecycleListener({
		nodeDetached: ({ child }) => {
			if (child !== parent) return;
			parent.removeChild(removed);
			parent.addChild(added);
			retained.name = "after";
			retained.visible = false;
			retained.position.x = 6;
			scene.add(parent);
		},
	});
	const world = new ECSWorld(scene);
	const parentEntity = world.getEntityByNode(parent);
	const retainedEntity = world.getEntityByNode(retained);
	const retainedTransform = world.getComponent(retainedEntity, "WorldTransform");
	const children = world.getComponent(parentEntity, "Hierarchy").children;
	scene.remove(parent);
	scene.updateWorldMatrices();
	const addedEntity = world.getEntityByNode(added);
	assert.equal(world.getEntityByNode(removed), null);
	assert.ok(addedEntity !== null);
	assert.equal(world.getEntityByNode(parent), parentEntity);
	assert.equal(world.getEntityByNode(retained), retainedEntity);
	assert.equal(world.query(["NodeRef"]).length, 4);
	assert.equal(world.getComponent(retainedEntity, "Name").value, "after");
	assert.equal(world.getComponent(retainedEntity, "Visibility").visible, false);
	assert.equal(world.getComponent(retainedEntity, "WorldTransform"), retainedTransform);
	assert.equal(retainedTransform.matrix.elements[0][3], 6);
	assert.deepEqual(Array.from(children), [retainedEntity, addedEntity]);
	world.destroy();
}

function testAbortedAttachmentCleansMovedProjectedDescendants() {
	for (const nested of [false, true]) {
		const scene = new Scene();
		const oldParent = new Node();
		const child = oldParent.addChild(new Node());
		const newParent = new Node();
		scene.add(oldParent);
		scene.addNodeLifecycleListener({
			nodeAttached: ({ child: attached }) => {
				if (attached === newParent) newParent.addChild(child);
				if ((!nested && attached === newParent) || (nested && attached === child)) {
					scene.remove(newParent);
					newParent.removeChild(child);
				}
			},
		});
		const world = new ECSWorld(scene);
		scene.add(newParent);
		assert.equal(world.getEntityByNode(newParent), null);
		assert.equal(world.getEntityByNode(child), null);
		assert.equal(world.query(["NodeRef"]).length, 2);
		assert.deepEqual(Array.from(world.getComponent(world.getEntityByNode(oldParent),
			"Hierarchy").children), []);
		world.destroy();
	}
}

for (const check of [
	run,
	testDerivedViewsRejectMutationAndStayLive,
	testReentrantMetadataPreservesFinalNodeState,
	testMetadataListenerCanDetachOrDestroyDuringWrite,
	testAliasedMetadataWritesAdvanceVersion,
	testEqualMatricesStillPublishAuthoredTransformChanges,
	testDirectMatrixResolutionRetainsPendingProjection,
	testReentrantTransformPublicationPreservesOtherListeners,
	testDetachedTransformChangesDoNotPublishEmptyBatches,
	testListenerErrorsDoNotSkipOtherWorlds,
	testReentrantAttachmentUsesCurrentOwnership,
	testReentrantDetachmentRemovesOriginalProjectedDescendants,
	testReentrantReattachmentPreservesProjection,
	testReattachedSubtreesReconcileRemovedAndAddedDescendants,
	testAbortedAttachmentCleansMovedProjectedDescendants,
]) {
	try {
		check();
	} catch (error) {
		console.error(check.name, error);
		process.exitCode = 1;
	}
}

if (!process.exitCode) console.log("Scene ECS projection tests passed");
