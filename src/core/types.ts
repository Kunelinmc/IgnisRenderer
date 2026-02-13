import type { IVector3 } from "../maths/types";
import type { RGBA } from "../utils/Color";

import type { Material } from "../materials/Material";

export interface IVertex extends IVector3 {
	u?: number;
	v?: number;
	normal?: IVector3 | null;
}

export interface IFace {
	vertices: IVertex[];
	material?: Material;
	color?: RGBA;
	normal?: IVector3;
	doubleSided?: boolean;
}

export interface DepthInfo {
	min: number;
	max: number;
	avg: number;
}

export interface ProjectedVertex extends IVector3 {
	w: number;
	u?: number;
	v?: number;
	normal?: IVector3 | null;
	world: IVertex;
}

export interface ProjectedFace extends IFace {
	projected: ProjectedVertex[];
	center: IVector3;
	depthInfo: DepthInfo;
	modelDepth: number;
}

export interface ProjectedPoint {
	x: number;
	y: number;
	z: number;
	depth: number;
	world: IVector3;
	iz: number;
}

export interface ITransform {
	position: IVector3;
	rotation: IVector3;
	scale: IVector3;
}

export interface BoundingSphere {
	center: IVector3;
	radius: number;
}

export interface BoundingBox {
	min: IVector3;
	max: IVector3;
}

export interface IModel {
	faces: IFace[];
	projectedFaces: ProjectedFace[];
	transform: ITransform;
	boundingSphere: BoundingSphere;
	boundingBox: BoundingBox;
	getFaceAtPoint(x: number, y: number): ProjectedFace | null;
}
