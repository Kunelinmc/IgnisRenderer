import type { Texture } from "../core/Texture";

export type TextureLike = Texture | null;
export type ShadingModel = "Flat" | "Gouraud" | "Phong" | "PBR" | "Unlit";
export type AlphaMode = "OPAQUE" | "MASK" | "BLEND";

export interface MirrorPlane {
	normal: { x: number; y: number; z: number };
	constant: number;
}

export interface MaterialParams {
	name?: string;
	type?: string;
	shading?: ShadingModel;
	opacity?: number;
	doubleSided?: boolean;
	wireframe?: boolean;
	alphaMode?: AlphaMode;
	alphaCutoff?: number;
	map?: TextureLike;
	reflectivity?: number;
	mirrorPlane?: MirrorPlane;
	vertexCode?: string;
	vertexJS?: any;
	fragmentCode?: string;
	fragmentJS?: any;
}

export class Material {
	public name: string;
	public type: string;
	public shading: ShadingModel;
	public opacity: number;
	public doubleSided: boolean;
	public wireframe: boolean;
	public alphaMode: AlphaMode;
	public alphaCutoff: number;
	public map: TextureLike;
	public reflectivity: number;
	public mirrorPlane: MirrorPlane | null;

	public vertexCode: string;
	public vertexJS: any;
	public fragmentCode: string;
	public fragmentJS: any;

	constructor(params: MaterialParams = {}) {
		this.name = params.name ?? "Untitled";
		this.type = params.type ?? "Basic";
		this.shading = params.shading ?? "Flat";
		this.opacity = params.opacity ?? 1;
		this.doubleSided = params.doubleSided ?? false;
		this.wireframe = params.wireframe ?? false;

		this.alphaMode = params.alphaMode ?? "OPAQUE";
		this.alphaCutoff = params.alphaCutoff ?? 0.5;
		this.map = params.map ?? null;

		this.reflectivity = params.reflectivity ?? 0;
		this.mirrorPlane = params.mirrorPlane ?? null;

		this.vertexCode = params.vertexCode ?? "";
		this.vertexJS = params.vertexJS ?? null;
		this.fragmentCode = params.fragmentCode ?? "";
		this.fragmentJS = params.fragmentJS ?? null;
	}
}
