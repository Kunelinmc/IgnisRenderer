import type { Texture } from "../core/Texture";

export type TextureLike = Texture | null;

export enum ShadingModel {
	Flat = "Flat",
	Gouraud = "Gouraud",
	Phong = "Phong",
	PBR = "PBR",
	Unlit = "Unlit",
}

export enum AlphaMode {
	Opaque = "OPAQUE",
	Mask = "MASK",
	Blend = "BLEND",
}

export enum CullMode {
	None = "none",
	Front = "front",
	Back = "back",
}

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
	cullMode?: CullMode;
	wireframe?: boolean;
	alphaMode?: AlphaMode;
	alphaCutoff?: number;
	depthWrite?: boolean;
	map?: TextureLike;
	reflectivity?: number;
	mirrorPlane?: MirrorPlane;
}

export class Material {
	public name: string;
	public type: string;
	public shading: ShadingModel;
	public opacity: number;
	public doubleSided: boolean;
	public cullMode: CullMode;
	public wireframe: boolean;
	public alphaMode: AlphaMode;
	public alphaCutoff: number;
	/**
	 * Controls whether opaque draws update the scene depth buffer after passing
	 * the depth test. Transparent draws remain depth-read-only.
	 */
	public depthWrite: boolean;
	public map: TextureLike;
	public reflectivity: number;
	public mirrorPlane: MirrorPlane | null;

	constructor(params: MaterialParams = {}) {
		this.name = params.name ?? "Untitled";
		this.type = params.type ?? "Basic";
		this.shading = params.shading ?? ShadingModel.Flat;
		this.opacity = params.opacity ?? 1;
		this.doubleSided = params.doubleSided ?? false;
		this.cullMode =
			params.cullMode ?? (this.doubleSided ? CullMode.None : CullMode.Back);
		this.wireframe = params.wireframe ?? false;

		this.alphaMode = params.alphaMode ?? AlphaMode.Opaque;
		this.alphaCutoff = params.alphaCutoff ?? 0.5;
		this.depthWrite = params.depthWrite ?? true;
		this.map = params.map ?? null;

		this.reflectivity = params.reflectivity ?? 0;
		this.mirrorPlane = params.mirrorPlane ?? null;
	}
}
