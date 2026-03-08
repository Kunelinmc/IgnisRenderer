import type { RGBA } from "../utils/Color";
import type { IVector3 } from "../maths/types";
import { GeometryBuilder, type GeometryFace } from "./GeometryBuilder";
import type {
	BoundingBox,
	BoundingSphere,
	IPrimitive,
	IVertex,
} from "../core/types";
import { IdGenerator } from "../utils/IdGenerator";

export type MeshVertex = GeometryFace["vertices"][number];
export type MeshFace = GeometryFace;

export class MeshAsset {
	public readonly id: string;
	public primitives: IPrimitive[];
	public boundingSphere: BoundingSphere;
	public boundingBox: BoundingBox;

	constructor(primitives: IPrimitive[] = []) {
		this.id = IdGenerator.nextId("mesh");
		this.primitives = primitives;
		this.boundingBox = GeometryBuilder.computeModelBoundingBox(this.primitives);
		this.boundingSphere = GeometryBuilder.computeModelBoundingSphere(
			this.primitives,
			this.boundingBox
		);
	}

	public static fromFaces(faces: MeshFace[] = []): MeshAsset {
		return new MeshAsset(GeometryBuilder.buildPrimitivesFromFaces(faces));
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
