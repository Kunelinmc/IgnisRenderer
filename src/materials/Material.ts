import { Texture } from "../core/Texture";

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
	private static _contentRevision = 0;
	private _revision = 0;
	private _renderSignature = "";
	public name: string;
	public type: string;
	public shading: ShadingModel;
	public opacity: number;
	/**
	 * Configures pipeline state culling (cullMode = None if true, Back if false)
	 * and flips shader normals on face back-sides for correct lighting.
	 */
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

	/** Monotonic revision of render-affecting material state. */
	public get revision(): number {
		this._refreshRevision(true);
		return this._revision;
	}

	/** Global revision used by on-demand renderers to observe material changes. */
	public static get contentRevision(): number {
		return Material._contentRevision;
	}

	/**
	 * Refreshes direct and nested render state and reports whether it changed.
	 *
	 * @internal Owned by renderer and backend material snapshot caches.
	 */
	public refreshRevision(): boolean {
		return this._refreshRevision(true);
	}

	/** @internal Returns the last refreshed revision without rescanning fields. */
	public _getRevisionInternal(): number {
		return this._revision;
	}

	private _refreshRevision(notify: boolean): boolean {
		const state = { a: 2166136261, b: 2246822519 };
		for (const key of Object.keys(this)) {
			if (key.startsWith("_") || key === "name") continue;
			mixString(state, key);
			mixRenderValue(state, (this as Material & Record<string, unknown>)[key], 0);
		}
		const extended = this as Material & {
			ior?: unknown;
			featureMask?: unknown;
			textureMask?: unknown;
			shaderRevision?: unknown;
			uniformValueRevision?: unknown;
		};
		for (const value of [
			extended.ior,
			extended.featureMask,
			extended.textureMask,
			extended.shaderRevision,
			extended.uniformValueRevision,
		]) mixRenderValue(state, value, 0);
		const signature = `${state.a}:${state.b}`;
		if (signature === this._renderSignature) return false;
		this._renderSignature = signature;
		if (!notify) return false;
		this._revision++;
		Material._contentRevision++;
		return true;
	}
}

interface RevisionHashState {
	a: number;
	b: number;
}

const REVISION_FLOAT_SCRATCH = new DataView(new ArrayBuffer(8));
const REVISION_PRIME = 16777619;
const MATERIAL_OBJECT_IDENTITIES = new WeakMap<object, number>();
let nextMaterialObjectIdentity = 1;

function mixRenderValue(state: RevisionHashState, value: unknown, depth: number): void {
	if (value === null || value === undefined) {
		mixUint(state, 0);
		return;
	}
	if (typeof value === "number") {
		REVISION_FLOAT_SCRATCH.setFloat64(0, value, true);
		mixUint(state, REVISION_FLOAT_SCRATCH.getUint32(0, true));
		mixUint(state, REVISION_FLOAT_SCRATCH.getUint32(4, true));
		return;
	}
	if (typeof value === "string" || typeof value === "boolean") {
		mixString(state, String(value));
		return;
	}
	if (typeof value !== "object") return;
	if (value instanceof Texture) {
		mixUint(state, getMaterialObjectIdentity(value));
		mixUint(state, value.version);
		mixUint(state, value.samplingRevision);
		return;
	}
	mixUint(state, getMaterialObjectIdentity(value));
	if (depth >= 2) return;
	if (ArrayBuffer.isView(value)) {
		mixUint(state, (value as ArrayBufferView).byteLength);
		return;
	}
	for (const key of Object.keys(value)) {
		if (key.startsWith("_")) continue;
		mixString(state, key);
		mixRenderValue(state, (value as Record<string, unknown>)[key], depth + 1);
	}
}

function getMaterialObjectIdentity(value: object): number {
	let identity = MATERIAL_OBJECT_IDENTITIES.get(value);
	if (identity !== undefined) return identity;
	identity = nextMaterialObjectIdentity++;
	MATERIAL_OBJECT_IDENTITIES.set(value, identity);
	return identity;
}

function mixString(state: RevisionHashState, value: string): void {
	mixUint(state, value.length);
	for (let index = 0; index < value.length; index++) mixUint(state, value.charCodeAt(index));
}

function mixUint(state: RevisionHashState, value: number): void {
	const normalized = value >>> 0;
	state.a = Math.imul((state.a ^ normalized) >>> 0, REVISION_PRIME) >>> 0;
	state.b = Math.imul((state.b ^ (normalized ^ 0x9e3779b9)) >>> 0, REVISION_PRIME) >>> 0;
}
