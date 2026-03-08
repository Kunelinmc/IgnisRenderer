import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import type { Texture } from "./Texture";
import { Node } from "./Node";
import type { BoundingSphere } from "./types";
import { MeshInstance } from "../meshes";
import type { SceneLight } from "../lights";
import { Camera } from "../cameras/Camera";
import { ParticleSystem } from "../particles";

export class Scene {
	public readonly root: Node;
	public skybox: Texture | null;

	private _version: number;

	constructor() {
		this.root = new Node({
			idPrefix: "scene",
			name: "sceneRoot",
		});
		this.skybox = null;
		this._version = 0;
	}

	public add<T extends Node>(node: T): T {
		this.root.addChild(node);
		this.invalidate();
		return node;
	}

	public remove(node: Node): boolean {
		const removed = node.parent ? node.parent.removeChild(node) : false;
		if (removed) {
			this.invalidate();
		}
		return removed;
	}

	public clear(): void {
		if (this.root.children.length === 0) return;
		for (const child of [...this.root.children]) {
			this.root.removeChild(child);
		}
		this.invalidate();
	}

	public contains(node: Node): boolean {
		if (node === this.root) return true;
		let found = false;
		this.traverse((current) => {
			if (current === node) {
				found = true;
			}
		});
		return found;
	}

	public traverse(visitor: (node: Node) => void): void {
		for (const child of this.root.children) {
			child.traverse(visitor);
		}
	}

	public getMeshInstances(): MeshInstance[] {
		return this._collectByType((node): node is MeshInstance => {
			return node instanceof MeshInstance;
		});
	}

	public getLights(): SceneLight[] {
		return this._collectByType((node): node is SceneLight => {
			return hasLightType(node);
		});
	}

	public getCameras(): Camera[] {
		return this._collectByType((node): node is Camera => {
			return node instanceof Camera;
		});
	}

	public getParticleSystems(): ParticleSystem[] {
		return this._collectByType((node): node is ParticleSystem => {
			return node instanceof ParticleSystem;
		});
	}

	public updateWorldMatrices(): void {
		this.root.updateWorldMatrix();
	}

	public invalidate(): void {
		this._version++;
	}

	public get version(): number {
		return this._version;
	}

	public getBounds(): BoundingSphere {
		let min: IVector3 = { x: Infinity, y: Infinity, z: Infinity };
		let max: IVector3 = { x: -Infinity, y: -Infinity, z: -Infinity };

		for (const meshInstance of this.getMeshInstances()) {
			if (meshInstance.visible === false) continue;
			const worldBounds = meshInstance.getWorldBoundingSphere();
			const center = worldBounds.center;
			const radius = worldBounds.radius;

			min.x = Math.min(min.x, center.x - radius);
			min.y = Math.min(min.y, center.y - radius);
			min.z = Math.min(min.z, center.z - radius);
			max.x = Math.max(max.x, center.x + radius);
			max.y = Math.max(max.y, center.y + radius);
			max.z = Math.max(max.z, center.z + radius);
		}

		if (min.x === Infinity) {
			return { center: { x: 0, y: 0, z: 0 }, radius: 100 };
		}

		const center: IVector3 = {
			x: (min.x + max.x) / 2,
			y: (min.y + max.y) / 2,
			z: (min.z + max.z) / 2,
		};
		const size: IVector3 = {
			x: max.x - min.x,
			y: max.y - min.y,
			z: max.z - min.z,
		};
		const radius =
			Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z) / 2;

		return { center, radius };
	}

	private _collectByType<T extends Node>(
		predicate: (node: Node) => node is T
	): T[] {
		const result: T[] = [];
		this.traverse((node) => {
			if (predicate(node)) {
				result.push(node);
			}
		});
		return result;
	}
}

function hasLightType(value: unknown): value is SceneLight {
	if (!value || typeof value !== "object") return false;
	return "type" in value && "intensity" in value && "color" in value;
}
