import type { Texture } from "../core/Texture";

export type TextureLike = Texture | null;
export type ShadingModel = "Flat" | "Gouraud" | "Phong" | "PBR" | "Unlit";
export type AlphaMode = "OPAQUE" | "MASK" | "BLEND";

export interface MirrorPlane {
	normal: { x: number; y: number; z: number };
	constant: number;
}

export interface MaterialParams {
	type?: string;
	shading?: ShadingModel;
	opacity?: number;
	doubleSided?: boolean;
	wireframe?: boolean;
	alphaMode?: AlphaMode;
	alphaCutoff?: number;
	map?: TextureLike;
	reflectivity?: number;
	fresnel?: boolean;
	mirrorPlane?: MirrorPlane;
}

export class Material {
	public type: string;
	public shading: ShadingModel;
	public opacity: number;
	public doubleSided: boolean;
	public wireframe: boolean;
	public alphaMode: AlphaMode;
	public alphaCutoff: number;
	public map: TextureLike;
	public reflectivity: number;
	public fresnel: boolean;
	public mirrorPlane: MirrorPlane | null;

	constructor(params: MaterialParams = {}) {
		this.type = params.type ?? "Basic";
		this.shading = params.shading ?? "Flat";
		this.opacity = params.opacity ?? 1;
		this.doubleSided = params.doubleSided ?? false;
		this.wireframe = params.wireframe ?? false;

		this.alphaMode = params.alphaMode ?? "OPAQUE";
		this.alphaCutoff = params.alphaCutoff ?? 0.5;
		this.map = params.map ?? null;

		this.reflectivity = params.reflectivity ?? 0;
		this.fresnel = params.fresnel ?? false;
		this.mirrorPlane = params.mirrorPlane ?? null;
	}
}
