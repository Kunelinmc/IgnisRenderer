import { Node, type NodeParams } from "../core/Node";
import type { BodyBinding } from "./types";

export interface PhysicsBodyNodeParams extends NodeParams {
	bodyBinding: BodyBinding;
}

export class PhysicsBodyNode extends Node {
	public bodyBinding: BodyBinding;

	constructor(params: PhysicsBodyNodeParams) {
		super({
			...params,
			idPrefix: "physicsBodyNode",
		});
		this.bodyBinding = cloneBodyBinding(params.bodyBinding);
	}

	protected override _createCloneInstance(): this {
		return new PhysicsBodyNode({
			bodyBinding: cloneBodyBinding(this.bodyBinding),
		}) as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.bodyBinding = cloneBodyBinding(this.bodyBinding);
	}
}

function cloneBodyBinding(bodyBinding: BodyBinding): BodyBinding {
	return {
		...bodyBinding,
		body: { ...bodyBinding.body },
		colliders: bodyBinding.colliders?.map((collider) => ({
			...collider,
		})),
	};
}
