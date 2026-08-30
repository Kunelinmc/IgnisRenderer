import type { GLTFAnimationBundle } from "../animation";
import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";

export class NodePrefab {
	public readonly root: Node;
	public readonly animationBundle: GLTFAnimationBundle | null;

	constructor(root: Node, animationBundle: GLTFAnimationBundle | null) {
		this.root = root;
		this.animationBundle = animationBundle;
	}

	public instantiate(scene: Scene): Node {
		const clonedRoot = this.root.clone(true);
		scene.add(clonedRoot);
		return clonedRoot;
	}
}
