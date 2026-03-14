import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";
import type { GLTFAnimationBundle } from "../animation";
import type { EntityId } from "./components";

export interface EntityPrefab {
	readonly root: Node;
	readonly animationBundle: GLTFAnimationBundle | null;
	instantiate(scene: Scene): { root: Node; rootEntity: EntityId | null };
}

export class NodeEntityPrefab implements EntityPrefab {
	public readonly root: Node;
	public readonly animationBundle: GLTFAnimationBundle | null;

	constructor(root: Node, animationBundle: GLTFAnimationBundle | null) {
		this.root = root;
		this.animationBundle = animationBundle;
	}

	public instantiate(scene: Scene): {
		root: Node;
		rootEntity: EntityId | null;
	} {
		const clonedRoot = this.root.clone(true);
		scene.add(clonedRoot);
		return {
			root: clonedRoot,
			rootEntity: scene.ecs.getEntityByNode(clonedRoot),
		};
	}
}
