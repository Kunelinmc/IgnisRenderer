import type { Vec4Tuple } from "../maths/Vector4";
import type { IVector3, IVector4 } from "../maths/types";
import type { RGBA } from "../foundation/Color";
import type { Material } from "../materials/Material";

export type PrimitiveDrawTopology =
	| "triangle-list"
	| "line-list"
	| "point-list";

export const DEFAULT_PRIMITIVE_DRAW_TOPOLOGY: PrimitiveDrawTopology =
	"triangle-list";

export interface IVertex extends IVector3 {
	u?: number;
	v?: number;
	u2?: number;
	v2?: number;
	u3?: number;
	v3?: number;
	u4?: number;
	v4?: number;
	normal?: IVector3 | null;
	tangent?: IVector4 | null;
	color?: RGBA;
	joints0?: Vec4Tuple;
	weights0?: Vec4Tuple;
	joints1?: Vec4Tuple;
	weights1?: Vec4Tuple;
}

export interface MorphTargetGeometry {
	positions?: Float32Array | null;
	normals?: Float32Array | null;
	tangents?: Float32Array | null;
}

export interface IPrimitiveGeometry {
	positions: Float32Array;
	normals?: Float32Array | null;
	tangents?: Float32Array | null;
	uv0?: Float32Array | null;
	uv1?: Float32Array | null;
	uv2?: Float32Array | null;
	uv3?: Float32Array | null;
	colors?: Float32Array | null;
	joints0?: Uint16Array | Uint32Array | null;
	weights0?: Float32Array | null;
	joints1?: Uint16Array | Uint32Array | null;
	weights1?: Float32Array | null;
	morphTargets?: MorphTargetGeometry[] | null;
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
	readonly geometry: IPrimitiveGeometry;
	readonly geometryVersion: number;
	topology?: PrimitiveDrawTopology;
	material: Material;
	readonly boundingSphere: BoundingSphere;
	readonly boundingBox: BoundingBox;
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
	material: Material;
	vertices: IVertex[];
	receiveShadows: boolean;
	normal?: IVector3;
}

export interface ProjectedVertex extends IVector3 {
	w: number;
	u?: number;
	v?: number;
	u2?: number;
	v2?: number;
	u3?: number;
	v3?: number;
	u4?: number;
	v4?: number;
	normal?: IVector3 | null;
	tangent?: IVector4 | null;
	world: IVertex;
	previousWorld?: IVertex;
	zView?: number;
}

export interface ProjectedFace extends PrimitiveFace {
	projected: ProjectedVertex[];
	center: IVector3;
	depthInfo: DepthInfo;
	/**
	 * Rasterizer winding classification used to orient double-sided normals.
	 * @internal Software renderer projection contract.
	 */
	frontFacing?: boolean;
}

export interface ProjectedPoint {
	x: number;
	y: number;
	z: number;
	depth: number;
	world: IVector3;
	iz: number;
}
