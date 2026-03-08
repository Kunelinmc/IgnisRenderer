import type { IVector3, IVector4 } from "../maths/types";
import type { RGBA } from "../utils/Color";
import type { Material } from "../materials/Material";

export interface IVertex extends IVector3 {
	u?: number;
	v?: number;
	u2?: number;
	v2?: number;
	normal?: IVector3 | null;
	tangent?: IVector4 | null;
	color?: RGBA;
}

export interface IPrimitiveGeometry {
	positions: Float32Array;
	normals?: Float32Array | null;
	tangents?: Float32Array | null;
	uv0?: Float32Array | null;
	uv1?: Float32Array | null;
	colors?: Float32Array | null;
	indices: Uint32Array;
}

export interface BoundingSphere {
	center: IVector3;
	radius: number;
}

export interface BoundingBox {
	min: IVector3;
	max: IVector3;
}

export interface IPrimitive {
	readonly id: string;
	geometry: IPrimitiveGeometry;
	material: Material;
	boundingSphere: BoundingSphere;
	boundingBox: BoundingBox;
	visible: boolean;
	castShadows: boolean;
	receiveShadows: boolean;
}

export interface DepthInfo {
	min: number;
	max: number;
	avg: number;
}

export interface PrimitiveFace {
	primitive: IPrimitive;
	material: Material;
	vertices: IVertex[];
	color?: RGBA;
	normal?: IVector3;
	doubleSided?: boolean;
}

export interface ProjectedVertex extends IVector3 {
	w: number;
	u?: number;
	v?: number;
	u2?: number;
	v2?: number;
	normal?: IVector3 | null;
	tangent?: IVector4 | null;
	world: IVertex;
	zView?: number;
}

export interface ProjectedFace extends PrimitiveFace {
	projected: ProjectedVertex[];
	center: IVector3;
	depthInfo: DepthInfo;
}

export interface ProjectedPoint {
	x: number;
	y: number;
	z: number;
	depth: number;
	world: IVector3;
	iz: number;
}
