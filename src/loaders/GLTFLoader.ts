import {
	SimpleModel,
	type ModelFace,
	type ModelVertex,
} from "../models/SimpleModel";
import { PBRMaterial, BasicMaterial } from "../materials";
import { Loader, type LoaderEvents } from "./Loader";
import { Matrix4 } from "../maths/Matrix4";
import type { Texture } from "../core/Texture";

export interface GLTFLoaderEvents extends LoaderEvents {
	load: [SimpleModel];
	parsestart: [];
	parseend: [SimpleModel];
}

const MAGIC_glTF = 0x46546c67;
const CHUNK_TYPE_JSON = 0x4e4f534a;
const CHUNK_TYPE_BIN = 0x004e4942;
const COMPONENT_TYPE_BYTE = 5120;
const COMPONENT_TYPE_UNSIGNED_BYTE = 5121;
const COMPONENT_TYPE_SHORT = 5122;
const COMPONENT_TYPE_UNSIGNED_SHORT = 5123;
const COMPONENT_TYPE_UNSIGNED_INT = 5125;
const COMPONENT_TYPE_FLOAT = 5126;
const TYPE_SCALAR = "SCALAR";
const TYPE_VEC2 = "VEC2";
const TYPE_VEC3 = "VEC3";
const TYPE_VEC4 = "VEC4";

/**
 * GLTFLoader handles both .glb (binary) and .gltf (JSON + external bins) formats.
 */
export class GLTFLoader extends Loader<GLTFLoaderEvents> {
	constructor() {
		super();
	}
	/**
	 * Loads a glTF or GLB model from a URL.
	 */
	public async load(url: string): Promise<SimpleModel> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const baseURL = url.substring(0, url.lastIndexOf("/") + 1);
			const model = await this.parse(buffer, baseURL);
			this.emit("load", model);
			return model;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}
	/**
	 * Parses glTF/GLB data.
	 */
	public async parse(
		data: ArrayBuffer,
		baseURL: string = ""
	): Promise<SimpleModel> {
		this.emit("parsestart");
		const dataView = new DataView(data);
		let json: any = null;
		let buffers: Uint8Array[] = [];
		// Check magic for GLB
		const magic = dataView.getUint32(0, true);
		if (magic === MAGIC_glTF) {
			// It's a GLB file
			const version = dataView.getUint32(4, true);
			if (version !== 2) throw new Error(`Unsupported GLB version: ${version}`);
			const length = dataView.getUint32(8, true);
			let offset = 12;
			while (offset < length) {
				const chunkLength = dataView.getUint32(offset, true);
				offset += 4;
				const chunkType = dataView.getUint32(offset, true);
				offset += 4;
				if (chunkType === CHUNK_TYPE_JSON) {
					const textDecoder = new TextDecoder("utf-8");
					const jsonBytes = new Uint8Array(data, offset, chunkLength);
					json = JSON.parse(textDecoder.decode(jsonBytes));
				} else if (chunkType === CHUNK_TYPE_BIN) {
					buffers[0] = new Uint8Array(data, offset, chunkLength);
				}
				// Align to 4-byte boundary for next chunk
				offset += (chunkLength + 3) & ~3;
			}
		} else {
			// It's a .gltf file (JSON)
			const textDecoder = new TextDecoder("utf-8");
			json = JSON.parse(textDecoder.decode(new Uint8Array(data)));
		}
		if (!json) throw new Error("Failed to parse glTF JSON");
		// Load external buffers if not already present (for .gltf)
		if (json.buffers) {
			const bufferPromises = json.buffers.map(
				async (bufferDef: any, i: number) => {
					if (buffers[i]) return; // Already loaded from GLB BIN chunk
					if (bufferDef.uri) {
						buffers[i] = await this._loadBuffer(bufferDef.uri, baseURL);
					}
				}
			);
			await Promise.all(bufferPromises);
		}
		// Pre-parse images and textures
		const images = await this.parseImages(json, buffers, baseURL);
		const textures = this.parseTextures(json, images);
		// Pre-parse materials
		const materials = this.parseMaterials(json, textures);
		let allFaces: ModelFace[] = [];
		const sceneIdx = json.scene !== undefined ? json.scene : 0;
		const scene = json.scenes && json.scenes[sceneIdx];
		if (scene && scene.nodes) {
			for (const nodeIdx of scene.nodes) {
				const faces = this.parseNode(
					json,
					nodeIdx,
					Matrix4.identity(),
					buffers,
					materials
				);
				allFaces = allFaces.concat(faces);
			}
		} else if (json.nodes) {
			for (let i = 0; i < json.nodes.length; i++) {
				const isChild = json.nodes.some(
					(n: any) => n.children && n.children.includes(i)
				);
				if (!isChild) {
					const faces = this.parseNode(
						json,
						i,
						Matrix4.identity(),
						buffers,
						materials
					);
					allFaces = allFaces.concat(faces);
				}
			}
		}
		const model = new SimpleModel(allFaces);
		this.emit("parseend", model);
		return model;
	}

	private async _loadBuffer(uri: string, baseURL: string): Promise<Uint8Array> {
		const url =
			uri.startsWith("data:") || uri.startsWith("http") ? uri : baseURL + uri;
		const response = await fetch(url);
		if (!response.ok) throw new Error(`Failed to load buffer from ${url}`);
		const arrayBuffer = await response.arrayBuffer();
		return new Uint8Array(arrayBuffer);
	}

	private _getMaterialTexture(
		texInfo: any,
		textures: (Texture | null)[]
	): Texture | null {
		if (texInfo === undefined) return null;
		const texIdx = texInfo.index;
		const tex = textures[texIdx];
		if (!tex) return null;

		const transform = texInfo.extensions?.KHR_texture_transform;
		if (!transform) return tex; // If no transform, we can just return the texture reference

		const cloned = tex.clone();
		if (transform.offset !== undefined) {
			cloned.offset.x = transform.offset[0];
			cloned.offset.y = transform.offset[1];
		}
		if (transform.scale !== undefined) {
			cloned.repeat.x = transform.scale[0];
			cloned.repeat.y = transform.scale[1];
		}
		if (transform.rotation !== undefined) {
			cloned.rotation = transform.rotation;
		}
		return cloned;
	}

	public parseMaterials(
		json: any,
		textures: (Texture | null)[] = []
	): PBRMaterial[] {
		if (!json.materials) return [];
		return json.materials.map((m: any) => {
			const pbr = m.pbrMetallicRoughness || {};
			const baseColor = pbr.baseColorFactor || [1, 1, 1, 1];
			const material = new PBRMaterial({
				albedo: {
					r: baseColor[0] * 255,
					g: baseColor[1] * 255,
					b: baseColor[2] * 255,
				},
				opacity: baseColor[3],
				roughness:
					pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 0.5,
				metalness: pbr.metallicFactor !== undefined ? pbr.metallicFactor : 0.0,
				emissive:
					m.emissiveFactor ?
						{
							r: m.emissiveFactor[0] * 255,
							g: m.emissiveFactor[1] * 255,
							b: m.emissiveFactor[2] * 255,
						}
					:	{ r: 0, g: 0, b: 0 },
				doubleSided: m.doubleSided || false,
			});
			if (pbr.baseColorTexture !== undefined) {
				const tex = this._getMaterialTexture(pbr.baseColorTexture, textures);
				if (tex) material.map = tex;
			}
			if (pbr.metallicRoughnessTexture !== undefined) {
				const tex = this._getMaterialTexture(
					pbr.metallicRoughnessTexture,
					textures
				);
				if (tex) material.metallicRoughnessMap = tex;
			}
			if (m.normalTexture !== undefined) {
				const tex = this._getMaterialTexture(m.normalTexture, textures);
				if (tex) material.normalMap = tex;
			}
			if (m.emissiveTexture !== undefined) {
				const tex = this._getMaterialTexture(m.emissiveTexture, textures);
				if (tex) material.emissiveMap = tex;
			}
			if (m.occlusionTexture !== undefined) {
				const tex = this._getMaterialTexture(m.occlusionTexture, textures);
				if (tex) material.occlusionMap = tex;
			}
			if (m.alphaMode !== undefined) (material as any).alphaMode = m.alphaMode;
			if (m.alphaCutoff !== undefined)
				(material as any).alphaCutoff = m.alphaCutoff;
			// KHR_materials_emissive_strength extension
			if (m.extensions?.KHR_materials_emissive_strength) {
				material.emissiveIntensity =
					m.extensions.KHR_materials_emissive_strength.emissiveStrength ?? 1.0;
			}
			return material;
		});
	}

	public async parseImages(
		json: any,
		buffers: Uint8Array[],
		baseURL: string
	): Promise<(Texture | null)[]> {
		if (!json.images) return [];
		const { TextureLoader } = await import("./TextureLoader");
		const loader = new TextureLoader();
		return Promise.all(
			json.images.map(async (img: any) => {
				if (img.bufferView !== undefined) {
					const bv = json.bufferViews[img.bufferView];
					const buf = buffers[bv.buffer || 0];
					const data = buf.subarray(
						bv.byteOffset || 0,
						(bv.byteOffset || 0) + bv.byteLength
					);
					// Use any cast to avoid SharedArrayBuffer/ArrayBuffer mismatch in some TS configs
					const blob = new Blob([data as any], {
						type: img.mimeType || "image/png",
					});
					return loader.loadFromBlob(blob);
				} else if (img.uri) {
					const url =
						img.uri.startsWith("data:") || img.uri.startsWith("http") ?
							img.uri
						:	baseURL + img.uri;
					return loader.load(url);
				}
				return null;
			})
		);
	}

	public parseTextures(
		json: any,
		images: (Texture | null)[]
	): (Texture | null)[] {
		if (!json.textures) return [];
		return json.textures.map((t: any) => {
			const texture = images[t.source];
			if (texture && t.sampler !== undefined) {
				const sampler = json.samplers[t.sampler];
				if (sampler.magFilter === 9728) texture.magFilter = "Nearest";
				else if (sampler.magFilter === 9729) texture.magFilter = "Linear";
				const minFilters: Record<number, string> = {
					9728: "Nearest",
					9729: "Linear",
					9984: "NearestMipmapNearest",
					9985: "LinearMipmapNearest",
					9986: "NearestMipmapLinear",
					9987: "LinearMipmapLinear",
				};
				if (sampler.minFilter !== undefined)
					texture.minFilter = minFilters[sampler.minFilter] || "Linear";
				const wrapModes: Record<number, "Repeat" | "Clamp" | "MirroredRepeat"> =
					{
						33071: "Clamp",
						10497: "Repeat",
						33648: "MirroredRepeat",
					};
				if (sampler.wrapS !== undefined)
					texture.wrapS = wrapModes[sampler.wrapS] || "Repeat";
				if (sampler.wrapT !== undefined)
					texture.wrapT = wrapModes[sampler.wrapT] || "Repeat";
			}
			return texture;
		});
	}

	public parseNode(
		json: any,
		nodeIdx: number,
		parentMatrix: Matrix4,
		buffers: Uint8Array[],
		materials: PBRMaterial[]
	): ModelFace[] {
		if (nodeIdx === undefined || !json.nodes || !json.nodes[nodeIdx]) return [];
		const node = json.nodes[nodeIdx];
		let localMatrix = Matrix4.identity();
		if (node.matrix) {
			localMatrix = Matrix4.fromArray(node.matrix);
		} else {
			if (node.translation)
				localMatrix = Matrix4.multiply(
					localMatrix,
					Matrix4.fromTranslation(node.translation)
				);
			if (node.rotation)
				localMatrix = Matrix4.multiply(
					localMatrix,
					Matrix4.fromQuaternion(node.rotation)
				);
			if (node.scale)
				localMatrix = Matrix4.multiply(
					localMatrix,
					Matrix4.fromScale(node.scale)
				);
		}
		const worldMatrix = Matrix4.multiply(parentMatrix, localMatrix);
		let faces: ModelFace[] = [];
		if (node.mesh !== undefined && json.meshes && json.meshes[node.mesh]) {
			const mesh = json.meshes[node.mesh];
			for (const primitive of mesh.primitives) {
				faces = faces.concat(
					this.parsePrimitive(json, primitive, buffers, materials, worldMatrix)
				);
			}
		}
		if (node.children) {
			for (const childIdx of node.children) {
				faces = faces.concat(
					this.parseNode(json, childIdx, worldMatrix, buffers, materials)
				);
			}
		}
		return faces;
	}

	public parsePrimitive(
		json: any,
		primitive: any,
		buffers: Uint8Array[],
		materials: PBRMaterial[],
		worldMatrix: Matrix4
	): ModelFace[] {
		const attrs = primitive.attributes;
		const material =
			primitive.material !== undefined && materials[primitive.material] ?
				materials[primitive.material]
			:	new BasicMaterial();
		if (attrs.POSITION === undefined) return [];
		const positions = this.getAccessorData(json, buffers, attrs.POSITION);
		const normals =
			attrs.NORMAL !== undefined ?
				this.getAccessorData(json, buffers, attrs.NORMAL)
			:	null;
		const tangents =
			attrs.TANGENT !== undefined ?
				this.getAccessorData(json, buffers, attrs.TANGENT)
			:	null;
		const uvs =
			attrs.TEXCOORD_0 !== undefined ?
				this.getAccessorData(json, buffers, attrs.TEXCOORD_0)
			:	null;
		const colors =
			attrs.COLOR_0 !== undefined ?
				this.getAccessorData(json, buffers, attrs.COLOR_0)
			:	null;
		const indices =
			primitive.indices !== undefined ?
				this.getAccessorData(json, buffers, primitive.indices)
			:	null;
		const faces: ModelFace[] = [];
		const faceCount = Math.floor(
			(indices ? indices.length : positions.length / 3) / 3
		);
		for (let i = 0; i < faceCount; i++) {
			const fv: ModelVertex[] = [];
			for (let j = 0; j < 3; j++) {
				const idx = indices ? indices[i * 3 + j] : i * 3 + j;
				const pos = {
					x: positions[idx * 3],
					y: positions[idx * 3 + 1],
					z: positions[idx * 3 + 2],
				};
				const tPos = Matrix4.transformPoint(worldMatrix, pos);
				const v: ModelVertex = { x: tPos.x!, y: tPos.y!, z: tPos.z! };
				if (uvs) {
					v.u = uvs[idx * 2];
					v.v = uvs[idx * 2 + 1];
				} else {
					v.u = 0;
					v.v = 0;
				}
				if (colors) {
					const acc = json.accessors[attrs.COLOR_0];
					const numComponents = acc.type === "VEC3" ? 3 : 4;
					v.color = {
						r: colors[idx * numComponents],
						g: colors[idx * numComponents + 1],
						b: colors[idx * numComponents + 2],
						a: numComponents === 4 ? colors[idx * numComponents + 3] : 1.0,
					};
				}
				if (normals) {
					const norm = {
						x: normals[idx * 3],
						y: normals[idx * 3 + 1],
						z: normals[idx * 3 + 2],
					};
					const normalMat = Matrix4.normalMatrix(worldMatrix);
					const tNorm = Matrix4.transformNormal(normalMat, norm);
					v.normal = { x: tNorm.x!, y: tNorm.y!, z: tNorm.z! };
				}
				if (tangents) {
					const tang = {
						x: tangents[idx * 4],
						y: tangents[idx * 4 + 1],
						z: tangents[idx * 4 + 2],
					};
					const w = tangents[idx * 4 + 3];
					const normalMat = Matrix4.normalMatrix(worldMatrix);
					const tTang = Matrix4.transformNormal(normalMat, tang);
					v.tangent = { x: tTang.x!, y: tTang.y!, z: tTang.z!, w };
				}
				fv.push(v);
			}
			faces.push({ vertices: fv, normal: undefined, material });
		}
		return faces;
	}

	public getAccessorData(json: any, buffers: Uint8Array[], index: number): any {
		const acc = json.accessors[index];
		const hasBaseBufferView = acc.bufferView !== undefined;
		let numComponents = (
			{
				[TYPE_SCALAR]: 1,
				[TYPE_VEC2]: 2,
				[TYPE_VEC3]: 3,
				[TYPE_VEC4]: 4,
			} as Record<string, number>
		)[acc.type];
		let elementSize = (
			{
				[COMPONENT_TYPE_FLOAT]: 4,
				[COMPONENT_TYPE_UNSIGNED_INT]: 4,
				[COMPONENT_TYPE_UNSIGNED_SHORT]: 2,
				[COMPONENT_TYPE_SHORT]: 2,
				[COMPONENT_TYPE_UNSIGNED_BYTE]: 1,
				[COMPONENT_TYPE_BYTE]: 1,
			} as Record<number, number>
		)[acc.componentType];
		let stride = numComponents * elementSize;
		const data = this.createTypedArray(
			acc.componentType,
			acc.count * numComponents
		);
		if (hasBaseBufferView) {
			const bv = json.bufferViews[acc.bufferView];
			const buf = buffers[bv.buffer || 0];
			const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
			stride = bv.byteStride || stride;

			// Fast path: Tightly packed and aligned
			const isAligned =
				(buf.byteOffset + byteOffset) % elementSize === 0 &&
				stride === numComponents * elementSize;

			if (isAligned && !acc.normalized && !acc.sparse) {
				const byteLength = acc.count * stride;
				const Constructor = this.getTypedArrayConstructor(acc.componentType);
				return new Constructor(
					buf.buffer,
					buf.byteOffset + byteOffset,
					acc.count * numComponents
				);
			}

			// Slow path: DataView with manual normalization
			const view = new DataView(
				buf.buffer,
				buf.byteOffset + byteOffset,
				acc.count * stride
			);
			for (let i = 0; i < acc.count; i++) {
				for (let j = 0; j < numComponents; j++) {
					const pos = i * stride + j * elementSize;
					let val = 0;
					switch (acc.componentType) {
						case COMPONENT_TYPE_FLOAT:
							val = view.getFloat32(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_INT:
							val = view.getUint32(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_SHORT:
							val = view.getUint16(pos, true);
							break;
						case COMPONENT_TYPE_SHORT:
							val = view.getInt16(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_BYTE:
							val = view.getUint8(pos);
							break;
						case COMPONENT_TYPE_BYTE:
							val = view.getInt8(pos);
							break;
					}
					if (acc.normalized) val = this.normalize(val, acc.componentType);
					(data as any)[i * numComponents + j] = val;
				}
			}
		}
		if (acc.sparse) {
			const s = acc.sparse;
			const idxBV = json.bufferViews[s.indices.bufferView];
			const valBV = json.bufferViews[s.values.bufferView];
			const idxBuf = buffers[idxBV.buffer || 0];
			const valBuf = buffers[valBV.buffer || 0];
			const idxSize = (
				{
					[COMPONENT_TYPE_UNSIGNED_INT]: 4,
					[COMPONENT_TYPE_UNSIGNED_SHORT]: 2,
					[COMPONENT_TYPE_UNSIGNED_BYTE]: 1,
				} as Record<number, number>
			)[s.indices.componentType];
			const idxView = new DataView(
				idxBuf.buffer,
				idxBuf.byteOffset +
					(idxBV.byteOffset || 0) +
					(s.indices.byteOffset || 0),
				s.count * idxSize
			);
			const valView = new DataView(
				valBuf.buffer,
				valBuf.byteOffset +
					(valBV.byteOffset || 0) +
					(s.values.byteOffset || 0),
				s.count * numComponents * elementSize
			);
			for (let i = 0; i < s.count; i++) {
				let idx = 0;
				if (s.indices.componentType === COMPONENT_TYPE_UNSIGNED_INT)
					idx = idxView.getUint32(i * idxSize, true);
				else if (s.indices.componentType === COMPONENT_TYPE_UNSIGNED_SHORT)
					idx = idxView.getUint16(i * idxSize, true);
				else idx = idxView.getUint8(i * idxSize);
				for (let j = 0; j < numComponents; j++) {
					const pos = (i * numComponents + j) * elementSize;
					let val = 0;
					switch (acc.componentType) {
						case COMPONENT_TYPE_FLOAT:
							val = valView.getFloat32(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_INT:
							val = valView.getUint32(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_SHORT:
							val = valView.getUint16(pos, true);
							break;
						case COMPONENT_TYPE_SHORT:
							val = valView.getInt16(pos, true);
							break;
						case COMPONENT_TYPE_UNSIGNED_BYTE:
							val = valView.getUint8(pos);
							break;
						case COMPONENT_TYPE_BYTE:
							val = valView.getInt8(pos);
							break;
					}
					if (acc.normalized) val = this.normalize(val, acc.componentType);
					(data as any)[idx * numComponents + j] = val;
				}
			}
		}
		return data;
	}

	public getTypedArrayConstructor(type: number): any {
		switch (type) {
			case COMPONENT_TYPE_FLOAT:
				return Float32Array;
			case COMPONENT_TYPE_UNSIGNED_INT:
				return Uint32Array;
			case COMPONENT_TYPE_UNSIGNED_SHORT:
				return Uint16Array;
			case COMPONENT_TYPE_SHORT:
				return Int16Array;
			case COMPONENT_TYPE_UNSIGNED_BYTE:
				return Uint8Array;
			case COMPONENT_TYPE_BYTE:
				return Int8Array;
			default:
				return Float32Array;
		}
	}

	public createTypedArray(
		type: number,
		length: number
	):
		| Float32Array
		| Uint32Array
		| Uint16Array
		| Int16Array
		| Uint8Array
		| Int8Array {
		const Constructor = this.getTypedArrayConstructor(type);
		return new Constructor(length);
	}

	public normalize(value: number, type: number): number {
		switch (type) {
			case COMPONENT_TYPE_UNSIGNED_BYTE:
				return value / 255.0;
			case COMPONENT_TYPE_BYTE:
				return Math.max(value / 127.0, -1.0);
			case COMPONENT_TYPE_UNSIGNED_SHORT:
				return value / 65535.0;
			case COMPONENT_TYPE_SHORT:
				return Math.max(value / 32767.0, -1.0);
			default:
				return value;
		}
	}
}
