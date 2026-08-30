import { WebGLCapabilityError } from "../../foundation/Error";
import type { Material } from "../../materials/Material";
import type { WebGLSceneMaterialVariant } from "../../shaders/webgl/sceneVariants";
import type { WebGLResolvedMaterialSnapshot } from "./WebGLMaterialSnapshotCache";
import {
	WEBGL_PBR_TEXTURE_SLOT_NAMES,
	type WebGLMaterialCommonState,
	type WebGLMaterialTextureState,
	type WebGLPBRMaterialState,
	type WebGLPBRTextureSlotName,
	type WebGLPhongMaterialState,
	type WebGLMaterialShadingFamily,
} from "./WebGLMaterialState";

export const WEBGL_MATERIAL_COMMON_BINDING = 0;
export const WEBGL_MATERIAL_LIGHTING_BINDING = 1;
export const WEBGL_MATERIAL_BUFFER_CACHE_MAX_ENTRIES = 16_384;

export interface WebGLMaterialBufferResources {
	readonly commonBuffer: WebGLBuffer;
	readonly lightingBuffer: WebGLBuffer | null;
	readonly revision: number;
	readonly layoutKey: string;
}

interface WebGLMaterialBufferEntry {
	commonBuffer: WebGLBuffer;
	lightingBuffer: WebGLBuffer | null;
	revision: number;
	layoutKey: string;
	shadingFamily: WebGLMaterialShadingFamily;
	commonByteLength: number;
	lightingByteLength: number;
}

/** @internal Owns revision-stable std140 buffers for built-in WebGL materials. */
export class WebGLMaterialBufferCache {
	private readonly _entries = new Map<Material, WebGLMaterialBufferEntry>();

	public constructor(
		private readonly _gl: WebGL2RenderingContext,
		private readonly _maxEntries = WEBGL_MATERIAL_BUFFER_CACHE_MAX_ENTRIES,
	) {}

	public resolve(
		material: Material,
		snapshot: WebGLResolvedMaterialSnapshot,
		variant: WebGLSceneMaterialVariant,
	): WebGLMaterialBufferResources {
		const layoutKey = getWebGLMaterialBufferLayoutKey(variant);
		let entry = this._entries.get(material);
		if (entry && entry.revision === snapshot.revision && entry.layoutKey === layoutKey) {
			this._touch(material, entry);
			return entry;
		}

		const commonData = packWebGLMaterialCommonState(snapshot.data.common, variant);
		const lightingData =
			snapshot.data.shadingFamily === "pbr" ?
				packWebGLPBRMaterialState(snapshot.data.lighting, variant)
			: snapshot.data.shadingFamily === "phong" ||
				snapshot.data.shadingFamily === "flat" ?
				packWebGLPhongMaterialState(snapshot.data.lighting)
			: null;
		if (!entry) {
			const commonBuffer = this._createBuffer("common");
			let lightingBuffer: WebGLBuffer | null = null;
			try {
				lightingBuffer = lightingData ?
					this._createBuffer(snapshot.data.shadingFamily) : null;
			} catch (error) {
				this._gl.deleteBuffer(commonBuffer);
				throw error;
			}
			entry = {
				commonBuffer,
				lightingBuffer,
				commonByteLength: 0,
				lightingByteLength: 0,
				revision: snapshot.revision,
				layoutKey,
				shadingFamily: snapshot.data.shadingFamily,
			};
		} else if (
			entry.shadingFamily !== snapshot.data.shadingFamily ||
			!!entry.lightingBuffer !== !!lightingData
		) {
			if (entry.lightingBuffer) this._gl.deleteBuffer(entry.lightingBuffer);
			entry.lightingBuffer = null;
			entry.lightingBuffer = lightingData ?
				this._createBuffer(snapshot.data.shadingFamily) : null;
			entry.lightingByteLength = 0;
		}
		entry.commonByteLength = this._upload(
			entry.commonBuffer,
			commonData,
			entry.commonByteLength,
		);
		if (entry.lightingBuffer && lightingData) {
			entry.lightingByteLength = this._upload(
				entry.lightingBuffer,
				lightingData,
				entry.lightingByteLength,
			);
		}
		entry.revision = snapshot.revision;
		entry.layoutKey = layoutKey;
		entry.shadingFamily = snapshot.data.shadingFamily;
		this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);
		this._entries.set(material, entry);
		this._touch(material, entry);
		this._trim();
		return entry;
	}

	public destroy(): void {
		for (const entry of this._entries.values()) {
			this._gl.deleteBuffer(entry.commonBuffer);
			if (entry.lightingBuffer) this._gl.deleteBuffer(entry.lightingBuffer);
		}
		this._entries.clear();
	}

	private _createBuffer(label: string): WebGLBuffer {
		const buffer = this._gl.createBuffer();
		if (!buffer) {
			throw new WebGLCapabilityError(
				"material-uniform-buffer-unavailable",
				`Failed to create the ${label} material buffer.`,
			);
		}
		return buffer;
	}

	private _upload(buffer: WebGLBuffer, data: Float32Array, currentSize: number): number {
		const gl = this._gl;
		gl.bindBuffer(gl.UNIFORM_BUFFER, buffer);
		if (currentSize !== data.byteLength) {
			gl.bufferData(gl.UNIFORM_BUFFER, data, gl.DYNAMIC_DRAW);
		} else {
			gl.bufferSubData(gl.UNIFORM_BUFFER, 0, data);
		}
		return data.byteLength;
	}

	private _touch(material: Material, entry: WebGLMaterialBufferEntry): void {
		this._entries.delete(material);
		this._entries.set(material, entry);
	}

	private _trim(): void {
		while (this._entries.size > this._maxEntries) {
			const oldest = this._entries.entries().next().value as
				| [Material, WebGLMaterialBufferEntry]
				| undefined;
			if (!oldest) break;
			this._entries.delete(oldest[0]);
			this._gl.deleteBuffer(oldest[1].commonBuffer);
			if (oldest[1].lightingBuffer) this._gl.deleteBuffer(oldest[1].lightingBuffer);
		}
	}
}

export function getWebGLMaterialBufferLayoutKey(
	variant: WebGLSceneMaterialVariant,
): string {
	const flags = [variant.baseMap, variant.emissiveMap];
	if (variant.model === "pbr") {
		for (const name of WEBGL_PBR_TEXTURE_SLOT_NAMES) flags.push(variant[name]);
	}
	return `${variant.model}:${flags.map((value) => value ? 1 : 0).join("")}`;
}

export function packWebGLMaterialCommonState(
	state: WebGLMaterialCommonState,
	variant: WebGLSceneMaterialVariant,
): Float32Array {
	const values: number[] = [];
	pushVec4(values, state.baseColor);
	pushVec4(values, state.emissive);
	pushVec4(values, state.alpha);
	pushVec4(values, state.renderParams);
	if (variant.baseMap) pushTextureTransform(values, state.baseMap);
	if (variant.emissiveMap) pushTextureTransform(values, state.emissiveMap);
	return new Float32Array(values);
}

export function packWebGLPBRMaterialState(
	state: WebGLPBRMaterialState,
	variant: WebGLSceneMaterialVariant,
): Float32Array {
	const values: number[] = [];
	pushVec4(values, state.pbr);
	pushVec4(values, state.specular);
	pushVec4(values, state.transmissionVolume);
	pushVec4(values, state.clearcoat);
	pushVec4(values, state.sheen);
	pushVec4(values, state.iridescence);
	pushVec4(values, state.attenuationColor);
	pushVec4(values, state.anisotropy);
	pushVec4(values, state.scales);
	for (const name of WEBGL_PBR_TEXTURE_SLOT_NAMES) {
		if (variant[name]) pushTextureTransform(values, state.textures[name]);
	}
	return new Float32Array(values);
}

export function packWebGLPhongMaterialState(
	state: WebGLPhongMaterialState,
): Float32Array {
	const values: number[] = [];
	pushVec4(values, state.specular);
	pushVec4(values, state.phong);
	pushVec4(values, state.phongAmbient);
	return new Float32Array(values);
}

function pushTextureTransform(
	values: number[],
	state: WebGLMaterialTextureState,
): void {
	pushVec4(values, state.transformA);
	pushVec4(values, state.transformB);
}

function pushVec4(values: number[], value: readonly number[]): void {
	values.push(value[0], value[1], value[2], value[3]);
}

export function getWebGLPBRTextureState(
	state: WebGLPBRMaterialState,
	name: WebGLPBRTextureSlotName,
): WebGLMaterialTextureState {
	return state.textures[name];
}
