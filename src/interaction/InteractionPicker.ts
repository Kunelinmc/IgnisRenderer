import type { Camera } from "../cameras/Camera";
import type { Node } from "../core/Node";
import type { Scene } from "../core/Scene";
import { MeshInstance } from "../meshes";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import type { PhysicsSystem } from "../physics/PhysicsSystem";
import type { SpatialRayHit } from "../spatial";
import type { Interactable, InteractableRegistry } from "./Interactable";
import { screenToWorldRay } from "./screenToWorldRay";
import type {
	InteractionDragRectState,
	InteractionHitResult,
	InteractionViewport,
} from "./types";

export type InteractionPickIntent = "hover" | "select";

export class InteractionPicker {
	private _scene: Scene | null = null;
	private _camera: Camera | null = null;
	private _physics: PhysicsSystem | null = null;
	private _maxRayDistance: number;
	private _bvhRayScratch: SpatialRayHit[] = [];
	private _interactables: InteractableRegistry;

	public constructor(
		maxRayDistance: number,
		interactables: InteractableRegistry
	) {
		this._maxRayDistance = maxRayDistance;
		this._interactables = interactables;
	}

	public attach(
		scene: Scene,
		camera: Camera,
		physics: PhysicsSystem | null
	): void {
		this._scene = scene;
		this._camera = camera;
		this._physics = physics;
	}

	public detach(): void {
		this._scene = null;
		this._camera = null;
		this._physics = null;
		this._bvhRayScratch.length = 0;
	}

	public pick(
		screenX: number,
		screenY: number,
		viewport: InteractionViewport,
		intent: InteractionPickIntent
	): InteractionHitResult | null {
		if (!this._scene || !this._camera) return null;
		this._scene.updateWorldMatrices();
		this._camera.updateMatrices();
		const ray = screenToWorldRay(this._camera, {
			screenX,
			screenY,
			viewportWidth: viewport.width,
			viewportHeight: viewport.height,
		});

		const physicsHit = this._pickPhysicsHit(ray.origin, ray.direction, intent);
		const bvhHit = this._pickBVHFallbackHit(ray.origin, ray.direction, intent);
		return chooseBetterHit(physicsHit, bvhHit);
	}

	public pickDragRect(
		rect: InteractionDragRectState,
		viewport: InteractionViewport
	): Node[] {
		if (!this._scene || !this._camera) return [];
		this._scene.updateWorldMatrices();
		this._camera.updateMatrices();
		const minX = Math.min(rect.startX, rect.endX);
		const minY = Math.min(rect.startY, rect.endY);
		const maxX = Math.max(rect.startX, rect.endX);
		const maxY = Math.max(rect.startY, rect.endY);
		if (maxX - minX < 2 || maxY - minY < 2) {
			return [];
		}

		const candidates: Array<{
			node: Node;
			priority: number;
			depth: number;
		}> = [];
		for (const meshInstance of this._scene.getMeshInstances()) {
			if (meshInstance.visible === false) continue;
			const interactable = this._resolveInteractable(meshInstance, "select");
			if (!interactable) continue;
			const sphere = meshInstance.getWorldBoundingSphere();
			const clip = screenProject(
				this._camera.viewProjectionMatrix,
				sphere.center,
				viewport.width,
				viewport.height
			);
			if (!clip || clip.w <= 0) continue;
			if (
				clip.screenX < minX ||
				clip.screenX > maxX ||
				clip.screenY < minY ||
				clip.screenY > maxY
			) {
				continue;
			}
			candidates.push({
				node: meshInstance,
				priority: normalizePriority(interactable.priority),
				depth: clip.depth,
			});
		}

		candidates.sort((a, b) => {
			if (a.priority !== b.priority) return b.priority - a.priority;
			if (a.depth !== b.depth) return a.depth - b.depth;
			return a.node.id.localeCompare(b.node.id);
		});
		return candidates.map((candidate) => candidate.node);
	}

	public isNodeSelectable(node: Node): boolean {
		return this._resolveInteractable(node, "select") !== null;
	}

	private _pickPhysicsHit(
		origin: IVector3,
		direction: IVector3,
		intent: InteractionPickIntent
	): InteractionHitResult | null {
		if (!this._physics) return null;
		const hits = this._physics.raycastAll({
			origin,
			direction,
			maxDistance: this._maxRayDistance,
			filter: {
				includeTriggers: false,
			},
		});
		if (hits.length === 0) return null;

		let best: InteractionHitResult | null = null;
		for (const hit of hits) {
			const node = this._physics.resolveHitNode(hit);
			if (!node) continue;
			const interactable = this._resolveInteractable(node, intent);
			if (!interactable) continue;
			const candidate: InteractionHitResult = {
				node,
				distance: hit.distance,
				priority: normalizePriority(interactable.priority),
				source: "physics",
			};
			best = chooseBetterHit(best, candidate);
		}
		return best;
	}

	private _pickBVHFallbackHit(
		origin: IVector3,
		direction: IVector3,
		intent: InteractionPickIntent
	): InteractionHitResult | null {
		if (!this._scene) return null;
		const meshInstances = this._scene
			.getMeshInstances()
			.filter((meshInstance) => meshInstance.visible !== false);
		if (meshInstances.length === 0) return null;

		const spatial = this._scene.rebuildSpatialIndex(meshInstances);
		const candidates = spatial.queryRayDetailedInto(
			origin,
			direction,
			this._bvhRayScratch,
			{
				includeInvisible: false,
				maxDistance: this._maxRayDistance,
				maxResults: 64,
			}
		);
		if (candidates.length === 0) return null;

		let best: InteractionHitResult | null = null;
		for (const candidate of candidates) {
			const meshInstance = candidate.meshInstance;
			const interactable = this._resolveInteractable(meshInstance, intent);
			if (!interactable) continue;
			let distance = candidate.distance;

			if (isAnimationDrivenMesh(meshInstance)) {
				const sphere = meshInstance.getWorldBoundingSphere();
				const sphereHit = intersectRaySphere(
					origin,
					direction,
					sphere.center,
					sphere.radius,
					this._maxRayDistance
				);
				if (sphereHit === null) continue;
				distance = sphereHit;
			} else {
				const triangleHit = intersectRayMeshTriangles(
					origin,
					direction,
					meshInstance,
					this._maxRayDistance
				);
				if (triangleHit === null) continue;
				distance = triangleHit;
			}

			best = chooseBetterHit(best, {
				node: meshInstance,
				distance,
				priority: normalizePriority(interactable.priority),
				source: "bvh",
			});
		}
		return best;
	}

	private _resolveInteractable(
		node: Node,
		intent: InteractionPickIntent
	): Interactable | null {
		if (!this._scene || !this._scene.contains(node)) return null;
		const interactable = this._interactables.get(node);
		if (!interactable || interactable.enabled === false) return null;
		if (intent === "hover" && interactable.hoverable === false) return null;
		if (intent === "select" && interactable.selectable === false) return null;
		return interactable;
	}
}

function chooseBetterHit(
	current: InteractionHitResult | null,
	candidate: InteractionHitResult | null
): InteractionHitResult | null {
	if (!current) return candidate;
	if (!candidate) return current;
	if (candidate.priority !== current.priority) {
		return candidate.priority > current.priority ? candidate : current;
	}
	if (candidate.distance !== current.distance) {
		return candidate.distance < current.distance ? candidate : current;
	}
	return candidate.node.id.localeCompare(current.node.id) < 0 ? candidate : current;
}

function normalizePriority(value: number | undefined): number {
	return Number.isFinite(value) ? Number(value) : 0;
}

function screenProject(
	viewProjection: { elements: number[][] },
	position: IVector3,
	width: number,
	height: number
): { screenX: number; screenY: number; depth: number; w: number } | null {
	const me = viewProjection.elements;
	const x = position.x;
	const y = position.y;
	const z = position.z;
	const clipX = me[0][0] * x + me[0][1] * y + me[0][2] * z + me[0][3];
	const clipY = me[1][0] * x + me[1][1] * y + me[1][2] * z + me[1][3];
	const clipZ = me[2][0] * x + me[2][1] * y + me[2][2] * z + me[2][3];
	const clipW = me[3][0] * x + me[3][1] * y + me[3][2] * z + me[3][3];
	if (Math.abs(clipW) < 1e-8) return null;
	const invW = 1 / clipW;
	const ndcX = clipX * invW;
	const ndcY = clipY * invW;
	const ndcZ = clipZ * invW;
	return {
		screenX: (ndcX * 0.5 + 0.5) * width,
		screenY: (0.5 - ndcY * 0.5) * height,
		depth: ndcZ,
		w: clipW,
	};
}

function isAnimationDrivenMesh(meshInstance: MeshInstance): boolean {
	if (meshInstance.skeleton) return true;
	for (const primitive of meshInstance.mesh.primitives) {
		if ((primitive.geometry.morphTargets?.length ?? 0) > 0) {
			return true;
		}
	}
	return false;
}

function intersectRaySphere(
	origin: IVector3,
	direction: IVector3,
	center: IVector3,
	radius: number,
	maxDistance: number
): number | null {
	const radiusClamped = Math.max(0.001, radius);
	const oc = Vector3.sub(origin, center);
	const b = Vector3.dot(oc, direction);
	const c = Vector3.dot(oc, oc) - radiusClamped * radiusClamped;
	if (c > 0 && b > 0) return null;
	const discriminant = b * b - c;
	if (discriminant < 0) return null;
	let distance = -b - Math.sqrt(discriminant);
	if (distance < 0) distance = 0;
	if (distance > maxDistance) return null;
	return distance;
}

function intersectRayMeshTriangles(
	origin: IVector3,
	direction: IVector3,
	meshInstance: MeshInstance,
	maxDistance: number
): number | null {
	let bestDistance = Infinity;
	for (const primitive of meshInstance.mesh.primitives) {
		if (primitive.visible === false) continue;
		const geometry = primitive.geometry;
		const positions = geometry.positions;
		const indices = geometry.indices;
		if (!positions || !indices || positions.length < 9 || indices.length < 3) {
			continue;
		}

		for (
			let triangleIndex = 0;
			triangleIndex + 2 < indices.length;
			triangleIndex += 3
		) {
			const i0 = indices[triangleIndex] * 3;
			const i1 = indices[triangleIndex + 1] * 3;
			const i2 = indices[triangleIndex + 2] * 3;
			const v0 = transformPosition(meshInstance, positions, i0);
			const v1 = transformPosition(meshInstance, positions, i1);
			const v2 = transformPosition(meshInstance, positions, i2);
			const distance = intersectRayTriangle(origin, direction, v0, v1, v2);
			if (distance === null) continue;
			if (distance > maxDistance) continue;
			if (distance < bestDistance) {
				bestDistance = distance;
			}
		}
	}
	return Number.isFinite(bestDistance) ? bestDistance : null;
}

function transformPosition(
	meshInstance: MeshInstance,
	positions: Float32Array,
	index: number
): IVector3 {
	const world = meshInstance.worldMatrix.elements;
	const x = positions[index];
	const y = positions[index + 1];
	const z = positions[index + 2];
	return {
		x: world[0][0] * x + world[0][1] * y + world[0][2] * z + world[0][3],
		y: world[1][0] * x + world[1][1] * y + world[1][2] * z + world[1][3],
		z: world[2][0] * x + world[2][1] * y + world[2][2] * z + world[2][3],
	};
}

function intersectRayTriangle(
	origin: IVector3,
	direction: IVector3,
	v0: IVector3,
	v1: IVector3,
	v2: IVector3
): number | null {
	const epsilon = 1e-8;
	const edge1 = Vector3.sub(v1, v0);
	const edge2 = Vector3.sub(v2, v0);
	const h = Vector3.cross(direction, edge2);
	const a = Vector3.dot(edge1, h);
	if (Math.abs(a) < epsilon) return null;

	const f = 1 / a;
	const s = Vector3.sub(origin, v0);
	const u = f * Vector3.dot(s, h);
	if (u < 0 || u > 1) return null;

	const q = Vector3.cross(s, edge1);
	const v = f * Vector3.dot(direction, q);
	if (v < 0 || u + v > 1) return null;

	const t = f * Vector3.dot(edge2, q);
	if (t < 0) return null;
	return t;
}
