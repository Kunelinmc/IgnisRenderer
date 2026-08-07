import type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IPrimitiveGeometry,
	IVertex,
} from "../core/types";
import type { RGBA } from "../foundation/Color";
import { IdGenerator } from "../foundation/IdGenerator";
import type { IVector3 } from "../maths/types";
import { GeometryBuilder, type GeometryFace } from "./GeometryBuilder";

export type MeshVertex = GeometryFace["vertices"][number];
export type MeshFace = GeometryFace;

interface PrimitiveBoundsState {
	geometryVersion: number;
	modelRadiusSq: number;
}

interface MutablePrimitive extends IPrimitive {
	geometry: IPrimitiveGeometry;
	geometryVersion: number;
}

const primitiveOwner = new WeakMap<IPrimitive, MeshAsset>();

export class MeshAsset {
	public readonly id: string;
	public defaultMorphWeights: Float32Array[];

	private _primitives: ReadonlyArray<IPrimitive>;
	private readonly _boundingSphere: BoundingSphere = {
		center: { x: 0, y: 0, z: 0 },
		radius: 0,
	};
	private readonly _boundingBox: BoundingBox = {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};
	private _boundsDirty = true;
	private _boundsVersion = 0;
	private _dirtyPrimitives = new Set<IPrimitive>();
	private _primitiveBoundsStates = new WeakMap<
		IPrimitive,
		PrimitiveBoundsState
	>();

	constructor(
		primitives: readonly IPrimitive[] = [],
		defaultMorphWeights: Float32Array[] = []
	) {
		this.id = IdGenerator.nextId("mesh");
		this._primitives = Object.freeze([]) as ReadonlyArray<IPrimitive>;
		this.defaultMorphWeights = defaultMorphWeights.map(
			(weights) => new Float32Array(weights)
		);
		this._replacePrimitiveSnapshot(primitives);
		this._refreshBoundsIfNeeded();
	}

	public get primitives(): ReadonlyArray<IPrimitive> {
		return this._primitives;
	}

	/**
	 * Monotonic token advanced by changes that may affect local mesh bounds.
	 * Consumers should compare the token rather than bounds object identity.
	 */
	public get boundsVersion(): number {
		return this._boundsVersion;
	}

	public get boundingSphere(): BoundingSphere {
		this._refreshBoundsIfNeeded();
		return this._boundingSphere;
	}

	public get boundingBox(): BoundingBox {
		this._refreshBoundsIfNeeded();
		return this._boundingBox;
	}

	/**
	 * Replaces the complete primitive snapshot and invalidates local bounds.
	 *
	 * @param primitives - Unique primitives not owned by another `MeshAsset`.
	 * @returns This asset for chaining.
	 * @throws `Error` if a primitive is duplicated or owned by another asset.
	 */
	public setPrimitives(primitives: readonly IPrimitive[]): this {
		this._replacePrimitiveSnapshot(primitives);
		return this;
	}

	/**
	 * Adds one primitive and takes exclusive ownership of it.
	 *
	 * @param primitive - A primitive not owned by any `MeshAsset`.
	 * @returns This asset for chaining.
	 * @throws `Error` if the primitive is already owned.
	 */
	public addPrimitive(primitive: IPrimitive): this {
		this._assertCanOwnPrimitive(primitive, false);
		primitiveOwner.set(primitive, this);
		this._primitives = Object.freeze([
			...this._primitives,
			primitive,
		]) as ReadonlyArray<IPrimitive>;
		this._invalidateBounds([primitive]);
		return this;
	}

	/**
	 * Replaces one primitive and transfers ownership to the new primitive.
	 *
	 * @param index - Index of the primitive to replace.
	 * @param primitive - A primitive not owned by any `MeshAsset`.
	 * @returns The primitive released from this asset.
	 * @throws `RangeError` if `index` is outside the primitive snapshot.
	 * @throws `Error` if the replacement primitive is already owned.
	 */
	public replacePrimitive(index: number, primitive: IPrimitive): IPrimitive {
		if (!Number.isInteger(index) || index < 0 || index >= this._primitives.length) {
			throw new RangeError(`MeshAsset primitive index ${index} is out of range`);
		}
		const previous = this._primitives[index];
		if (previous === primitive) return previous;
		this._assertCanOwnPrimitive(primitive, false);

		const next = this._primitives.slice();
		next[index] = primitive;
		primitiveOwner.delete(previous);
		primitiveOwner.set(primitive, this);
		this._primitives = Object.freeze(next);
		this._invalidateBounds([primitive]);
		return previous;
	}

	/**
	 * Removes and releases a primitive owned by this asset.
	 *
	 * @param primitive - Primitive to remove.
	 * @returns Whether the primitive belonged to this asset and was removed.
	 */
	public removePrimitive(primitive: IPrimitive): boolean {
		const index = this._primitives.indexOf(primitive);
		if (index < 0) return false;
		const next = this._primitives.slice();
		next.splice(index, 1);
		primitiveOwner.delete(primitive);
		this._primitives = Object.freeze(next);
		this._invalidateBounds();
		return true;
	}

	/**
	 * Replaces owned primitive geometry and invalidates local bounds.
	 *
	 * @param primitive - Primitive owned by this asset.
	 * @param geometry - Replacement geometry.
	 * @returns This asset for chaining.
	 * @throws `Error` if this asset does not own `primitive`.
	 */
	public setPrimitiveGeometry(
		primitive: IPrimitive,
		geometry: IPrimitiveGeometry
	): this {
		this._assertOwnsPrimitive(primitive);
		const mutable = primitive as MutablePrimitive;
		mutable.geometry = geometry;
		mutable.geometryVersion = nextGeometryVersion(primitive.geometryVersion);
		this._invalidateBounds([primitive]);
		return this;
	}

	/**
	 * Reports in-place edits to an owned primitive's geometry buffers.
	 *
	 * @param primitive - Primitive owned by this asset.
	 * @returns This asset for chaining.
	 * @throws `Error` if this asset does not own `primitive`.
	 */
	public markPrimitiveGeometryDirty(primitive: IPrimitive): this {
		this._assertOwnsPrimitive(primitive);
		const mutable = primitive as MutablePrimitive;
		mutable.geometryVersion = nextGeometryVersion(primitive.geometryVersion);
		this._invalidateBounds([primitive]);
		return this;
	}

	public static fromFaces(faces: MeshFace[] = []): MeshAsset {
		return new MeshAsset(GeometryBuilder.buildPrimitivesFromFaces(faces));
	}

	private _replacePrimitiveSnapshot(primitives: readonly IPrimitive[]): void {
		const previous = this._primitives;
		const previousSet = new Set(previous);
		const unique = new Set<IPrimitive>();
		for (const primitive of primitives) {
			if (unique.has(primitive)) {
				throw new Error("MeshAsset cannot contain the same primitive twice");
			}
			unique.add(primitive);
			this._assertCanOwnPrimitive(primitive, true);
		}

		for (const primitive of previous) {
			if (!unique.has(primitive)) primitiveOwner.delete(primitive);
		}
		for (const primitive of primitives) {
			primitiveOwner.set(primitive, this);
		}

		this._primitives = Object.freeze(primitives.slice());
		const dirty: IPrimitive[] = [];
		for (const primitive of primitives) {
			if (!previousSet.has(primitive)) dirty.push(primitive);
		}
		this._invalidateBounds(dirty);
	}

	private _assertCanOwnPrimitive(
		primitive: IPrimitive,
		allowCurrentOwner: boolean
	): void {
		const owner = primitiveOwner.get(primitive);
		if (!owner) return;
		if (allowCurrentOwner && owner === this) return;
		throw new Error(
			owner === this ?
				"MeshAsset already owns this primitive"
			:	"Primitive is already owned by another MeshAsset"
		);
	}

	private _assertOwnsPrimitive(primitive: IPrimitive): void {
		if (primitiveOwner.get(primitive) !== this) {
			throw new Error("MeshAsset does not own this primitive");
		}
	}

	private _invalidateBounds(primitives: readonly IPrimitive[] = []): void {
		this._boundsDirty = true;
		this._boundsVersion++;
		for (const primitive of primitives) this._dirtyPrimitives.add(primitive);
	}

	private _refreshBoundsIfNeeded(): void {
		if (!this._boundsDirty) return;
		this._recomputeBounds();
	}

	private _recomputeBounds(): void {
		if (this._primitives.length === 0) {
			setBoxToZero(this._boundingBox);
			setSphereToZero(this._boundingSphere);
			this._finishBoundsRefresh();
			return;
		}

		for (const primitive of this._dirtyPrimitives) {
			computePrimitiveBoundingBoxInto(primitive.geometry, primitive.boundingBox);
		}
		computeModelBoundingBoxInto(this._primitives, this._boundingBox);

		const centerX = (this._boundingBox.min.x + this._boundingBox.max.x) * 0.5;
		const centerY = (this._boundingBox.min.y + this._boundingBox.max.y) * 0.5;
		const centerZ = (this._boundingBox.min.z + this._boundingBox.max.z) * 0.5;
		const centerChanged =
			this._boundingSphere.center.x !== centerX ||
			this._boundingSphere.center.y !== centerY ||
			this._boundingSphere.center.z !== centerZ;

		this._boundingSphere.center.x = centerX;
		this._boundingSphere.center.y = centerY;
		this._boundingSphere.center.z = centerZ;

		if (centerChanged) {
			for (const primitive of this._primitives) {
				this._refreshPrimitiveRadii(primitive, centerX, centerY, centerZ, true);
			}
		} else {
			for (const primitive of this._dirtyPrimitives) {
				this._refreshPrimitiveRadii(primitive, centerX, centerY, centerZ, false);
			}
		}

		let modelRadiusSq = 0;
		for (const primitive of this._primitives) {
			const state = this._primitiveBoundsStates.get(primitive);
			if (state && state.modelRadiusSq > modelRadiusSq) {
				modelRadiusSq = state.modelRadiusSq;
			}
		}
		this._boundingSphere.radius = Math.sqrt(modelRadiusSq);
		this._finishBoundsRefresh();
	}

	private _refreshPrimitiveRadii(
		primitive: IPrimitive,
		modelCenterX: number,
		modelCenterY: number,
		modelCenterZ: number,
		refreshModelRadius: boolean
	): void {
		const dirty = this._dirtyPrimitives.has(primitive);
		let state = this._primitiveBoundsStates.get(primitive);
		if (!state) {
			state = { geometryVersion: primitive.geometryVersion, modelRadiusSq: 0 };
			this._primitiveBoundsStates.set(primitive, state);
		}

		const box = primitive.boundingBox;
		const primitiveCenterX = (box.min.x + box.max.x) * 0.5;
		const primitiveCenterY = (box.min.y + box.max.y) * 0.5;
		const primitiveCenterZ = (box.min.z + box.max.z) * 0.5;
		if (dirty) {
			primitive.boundingSphere.center.x = primitiveCenterX;
			primitive.boundingSphere.center.y = primitiveCenterY;
			primitive.boundingSphere.center.z = primitiveCenterZ;
		}

		let primitiveRadiusSq = 0;
		let modelRadiusSq = refreshModelRadius ? 0 : state.modelRadiusSq;
		const positions = primitive.geometry.positions;
		for (let i = 0; i < positions.length; i += 3) {
			const x = positions[i];
			const y = positions[i + 1];
			const z = positions[i + 2];
			if (dirty) {
				const dx = x - primitiveCenterX;
				const dy = y - primitiveCenterY;
				const dz = z - primitiveCenterZ;
				const distanceSq = dx * dx + dy * dy + dz * dz;
				if (distanceSq > primitiveRadiusSq) primitiveRadiusSq = distanceSq;
			}
			if (dirty || refreshModelRadius) {
				const dx = x - modelCenterX;
				const dy = y - modelCenterY;
				const dz = z - modelCenterZ;
				const distanceSq = dx * dx + dy * dy + dz * dz;
				if (distanceSq > modelRadiusSq) modelRadiusSq = distanceSq;
			}
		}

		if (dirty) primitive.boundingSphere.radius = Math.sqrt(primitiveRadiusSq);
		state.geometryVersion = primitive.geometryVersion;
		state.modelRadiusSq = modelRadiusSq;
	}

	private _finishBoundsRefresh(): void {
		this._boundsDirty = false;
		this._dirtyPrimitives.clear();
	}
}

function nextGeometryVersion(version: number): number {
	return Number.isFinite(version) ? version + 1 : 1;
}

function computePrimitiveBoundingBoxInto(
	geometry: IPrimitiveGeometry,
	target: BoundingBox
): void {
	const positions = geometry.positions;
	if (positions.length === 0) {
		setBoxToZero(target);
		return;
	}

	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	for (let i = 0; i < positions.length; i += 3) {
		const x = positions[i];
		const y = positions[i + 1];
		const z = positions[i + 2];
		if (x < minX) minX = x;
		if (y < minY) minY = y;
		if (z < minZ) minZ = z;
		if (x > maxX) maxX = x;
		if (y > maxY) maxY = y;
		if (z > maxZ) maxZ = z;
	}
	target.min.x = minX;
	target.min.y = minY;
	target.min.z = minZ;
	target.max.x = maxX;
	target.max.y = maxY;
	target.max.z = maxZ;
}

function computeModelBoundingBoxInto(
	primitives: ReadonlyArray<IPrimitive>,
	target: BoundingBox
): void {
	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;
	for (const primitive of primitives) {
		const box = primitive.boundingBox;
		if (box.min.x < minX) minX = box.min.x;
		if (box.min.y < minY) minY = box.min.y;
		if (box.min.z < minZ) minZ = box.min.z;
		if (box.max.x > maxX) maxX = box.max.x;
		if (box.max.y > maxY) maxY = box.max.y;
		if (box.max.z > maxZ) maxZ = box.max.z;
	}
	target.min.x = minX;
	target.min.y = minY;
	target.min.z = minZ;
	target.max.x = maxX;
	target.max.y = maxY;
	target.max.z = maxZ;
}

function setBoxToZero(target: BoundingBox): void {
	target.min.x = 0;
	target.min.y = 0;
	target.min.z = 0;
	target.max.x = 0;
	target.max.y = 0;
	target.max.z = 0;
}

function setSphereToZero(target: BoundingSphere): void {
	target.center.x = 0;
	target.center.y = 0;
	target.center.z = 0;
	target.radius = 0;
}

export type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IVertex,
	IVector3,
	RGBA,
};
