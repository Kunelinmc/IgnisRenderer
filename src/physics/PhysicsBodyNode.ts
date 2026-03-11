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
		this.bodyBinding = {
			...params.bodyBinding,
			body: { ...params.bodyBinding.body },
			colliders: params.bodyBinding.colliders?.map((collider) => ({
				...collider,
			})),
		};
	}
}
