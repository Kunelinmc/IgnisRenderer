import type { RGBA } from "../foundation/Color";
import type { IVector3 } from "../maths/types";
import { GeometryBuilder, type GeometryFace } from "./GeometryBuilder";
import type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IVertex,
} from "../core/types";
import { IdGenerator } from "../foundation/IdGenerator";

export type MeshVertex = GeometryFace["vertices"][number];
export type MeshFace = GeometryFace;

export class MeshAsset {
	public readonly id: string;
	private _primitives: IPrimitive[];
	public defaultMorphWeights: Float32Array[];
	private _boundingSphere: BoundingSphere = {
		center: { x: 0, y: 0, z: 0 },
		radius: 0,
	};
	private _boundingBox: BoundingBox = {
		min: { x: 0, y: 0, z: 0 },
		max: { x: 0, y: 0, z: 0 },
	};
	private _boundsDirty = true;
	private _boundsPrimitiveSnapshot: IPrimitive[] = [];
	private _primitiveBoundsVersions = new WeakMap<IPrimitive, number>();

	constructor(
		primitives: IPrimitive[] = [],
		defaultMorphWeights: Float32Array[] = []
	) {
		this.id = IdGenerator.nextId("mesh");
		this._primitives = this._createReactivePrimitives(primitives);
		this.defaultMorphWeights = defaultMorphWeights.map(
			(weights) => new Float32Array(weights)
		);
		this._recomputeBounds();
	}

	public get primitives(): IPrimitive[] {
		return this._primitives;
	}

	public set primitives(primitives: IPrimitive[]) {
		this._primitives = this._createReactivePrimitives(primitives);
		this._invalidateBounds();
	}

	public get boundingSphere(): BoundingSphere {
		this._refreshBoundsIfNeeded();
		return this._boundingSphere;
	}

	public set boundingSphere(bounds: BoundingSphere) {
		this._boundingSphere = bounds;
		this._syncBoundsSnapshot();
	}

	public get boundingBox(): BoundingBox {
		this._refreshBoundsIfNeeded();
		return this._boundingBox;
	}

	public set boundingBox(bounds: BoundingBox) {
		this._boundingBox = bounds;
		this._syncBoundsSnapshot();
	}

	public static fromFaces(faces: MeshFace[] = []): MeshAsset {
		return new MeshAsset(GeometryBuilder.buildPrimitivesFromFaces(faces));
	}

	private _createReactivePrimitives(primitives: IPrimitive[]): IPrimitive[] {
		return new Proxy(primitives, {
			set: (target, property, value, receiver) => {
				const previous = Reflect.get(target, property, receiver);
				const updated = Reflect.set(target, property, value, receiver);
				if (updated && previous !== value) {
					this._invalidateBounds();
				}
				return updated;
			},
			deleteProperty: (target, property) => {
				const hadProperty = Reflect.has(target, property);
				const deleted = Reflect.deleteProperty(target, property);
				if (deleted && hadProperty) {
					this._invalidateBounds();
				}
				return deleted;
			},
		});
	}

	private _invalidateBounds(): void {
		this._boundsDirty = true;
	}

	private _refreshBoundsIfNeeded(): void {
		if (!this._boundsDirty && !this._havePrimitiveBoundsChanged()) {
			return;
		}
		this._recomputeBounds();
	}

	private _havePrimitiveBoundsChanged(): boolean {
		if (this._boundsPrimitiveSnapshot.length !== this._primitives.length) {
			return true;
		}

		for (let i = 0; i < this._primitives.length; i++) {
			const primitive = this._primitives[i];
			if (this._boundsPrimitiveSnapshot[i] !== primitive) {
				return true;
			}
			const boundsVersion = this._primitiveBoundsVersions.get(primitive);
			if (boundsVersion !== primitive.geometryVersion) {
				return true;
			}
		}

		return false;
	}

	private _recomputeBounds(): void {
		for (const primitive of this._primitives) {
			const boundingBox = GeometryBuilder.computeBoundingBox(primitive.geometry);
			primitive.boundingBox = boundingBox;
			primitive.boundingSphere = GeometryBuilder.computeBoundingSphere(
				primitive.geometry,
				boundingBox
			);
		}

		this._boundingBox = GeometryBuilder.computeModelBoundingBox(this._primitives);
		this._boundingSphere = GeometryBuilder.computeModelBoundingSphere(
			this._primitives,
			this._boundingBox
		);
		this._syncBoundsSnapshot();
	}

	private _syncBoundsSnapshot(): void {
		this._boundsDirty = false;
		this._boundsPrimitiveSnapshot = this._primitives.slice();
		this._primitiveBoundsVersions = new WeakMap<IPrimitive, number>();
		for (const primitive of this._primitives) {
			this._primitiveBoundsVersions.set(primitive, primitive.geometryVersion);
		}
	}
}

export type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IVertex,
	IVector3,
	RGBA,
};
