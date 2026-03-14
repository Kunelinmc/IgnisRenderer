import { Loader, type LoaderEvents } from "./Loader";
import { PBRMaterial, UnlitMaterial, type Material } from "../materials";
import type { Texture } from "../core/Texture";
import { Node } from "../core/Node";
import { Matrix4 } from "../maths/Matrix4";
import { Quaternion } from "../maths/Quaternion";
import { Camera, CameraType } from "../cameras/Camera";
import { OrthographicCamera } from "../cameras/OrthographicCamera";
import {
	AmbientLight,
	DirectionalLight,
	PointLight,
	SpotLight,
	type SceneLight,
} from "../lights";
import { MeshAsset, MeshInstance } from "../meshes";
import { GeometryBuilder } from "../meshes/GeometryBuilder";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type {
	IPrimitive,
	IPrimitiveGeometry,
	MorphTargetGeometry,
	PrimitiveDrawTopology,
} from "../core/types";
import { IdGenerator } from "../utils/IdGenerator";
import {
	AnimationClip,
	type GLTFAnimationBundle,
	KeyframeTrack,
	Skeleton,
} from "../animation";
import { NodeEntityPrefab } from "../ecs";
import type { EntityPrefab } from "../ecs";

export interface GLTFLoaderEvents extends LoaderEvents {
	load: [Node];
	parsestart: [];
	parseend: [Node];
	loadprefab: [EntityPrefab];
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
const TYPE_MAT2 = "MAT2";
const TYPE_MAT3 = "MAT3";
const TYPE_MAT4 = "MAT4";
const GLTF_MODE_POINTS = 0;
const GLTF_MODE_LINES = 1;
const GLTF_MODE_LINE_LOOP = 2;
const GLTF_MODE_LINE_STRIP = 3;
const GLTF_MODE_TRIANGLES = 4;
const GLTF_MODE_TRIANGLE_STRIP = 5;
const GLTF_MODE_TRIANGLE_FAN = 6;

interface GLTFParseContext {
	nodeByIndex: Map<number, Node>;
	nodePathByIndex: Map<number, string>;
	pathToNode: Map<string, Node>;
	meshInstanceByNodeIndex: Map<number, MeshInstance>;
	pathToMeshInstance: Map<string, MeshInstance>;
	pendingSkinByInstance: Map<MeshInstance, number>;
	meshCache: Map<number, MeshAsset | null>;
	skeletonBySkinIndex: Map<number, Skeleton>;
}

/**
 * GLTFLoader handles both .glb (binary) and .gltf (JSON + external bins) formats.
 */
export class GLTFLoader extends Loader<GLTFLoaderEvents> {
	private _lastAnimationBundle: GLTFAnimationBundle | null = null;

	constructor() {
		super();
	}

	public getLastAnimationBundle(): GLTFAnimationBundle | null {
		return this._lastAnimationBundle;
	}

	public clearLastAnimationBundle(): void {
		this._lastAnimationBundle = null;
	}
	/**
	 * Loads a glTF or GLB model from a URL.
	 */
	public async load(url: string): Promise<Node> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const baseURL = url.substring(0, url.lastIndexOf("/") + 1);
			const root = await this.parse(buffer, baseURL);
			this.emit("load", root);
			return root;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}

	public async loadPrefab(url: string): Promise<EntityPrefab> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const baseURL = url.substring(0, url.lastIndexOf("/") + 1);
			const prefab = await this.parsePrefab(buffer, baseURL);
			this.emit("loadprefab", prefab);
			return prefab;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}
	/**
	 * Parses glTF/GLB data.
	 */
	public async parse(data: ArrayBuffer, baseURL: string = ""): Promise<Node> {
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
		this._lastAnimationBundle = null;

		const context: GLTFParseContext = {
			nodeByIndex: new Map(),
			nodePathByIndex: new Map(),
			pathToNode: new Map(),
			meshInstanceByNodeIndex: new Map(),
			pathToMeshInstance: new Map(),
			pendingSkinByInstance: new Map(),
			meshCache: new Map(),
			skeletonBySkinIndex: new Map(),
		};
		const root = new Node({
			idPrefix: "node",
			name: "gltfRoot",
		});
		const lights = json.extensions?.KHR_lights_punctual?.lights ?? [];

		const sceneIdx = json.scene !== undefined ? json.scene : 0;
		const scene = json.scenes && json.scenes[sceneIdx];
		if (scene && scene.nodes) {
			for (const nodeIdx of scene.nodes) {
				const node = this.parseNodeTree(
					json,
					nodeIdx,
					buffers,
					materials,
					lights,
					context,
					"/gltfRoot"
				);
				root.addChild(node);
			}
		} else if (json.nodes) {
			for (let i = 0; i < json.nodes.length; i++) {
				const isChild = json.nodes.some(
					(n: any) => n.children && n.children.includes(i)
				);
				if (!isChild) {
					const node = this.parseNodeTree(
						json,
						i,
						buffers,
						materials,
						lights,
						context,
						"/gltfRoot"
					);
					root.addChild(node);
				}
			}
		}

		const skeletons = this.parseSkins(json, buffers, context);
		const clips = this.parseAnimations(json, buffers, context);
		this._lastAnimationBundle = {
			clips,
			skeletons,
			morphBindings: Array.from(context.pathToMeshInstance.entries())
				.filter(
					([, instance]) =>
						instance.skeleton !== undefined ||
						instance.morphWeights.some((weights) => weights.length > 0)
				)
				.map(([path, instance]) => ({
					path,
					instance,
					targetCount: instance.morphWeights[0]?.length ?? 0,
				})),
			nodePathMap: Object.fromEntries(
				Array.from(context.pathToNode.entries()).map(([path, node]) => [
					path,
					node.id,
				])
			),
		};

		this.emit("parseend", root);
		return root;
	}

	public async parsePrefab(
		data: ArrayBuffer,
		baseURL: string = ""
	): Promise<EntityPrefab> {
		const root = await this.parse(data, baseURL);
		return new NodeEntityPrefab(root, this._lastAnimationBundle);
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
		textures: (Texture | null)[],
		colorSpace?: Texture["colorSpace"]
	): Texture | null {
		if (texInfo === undefined) return null;
		const texIdx = texInfo.index;
		const tex = textures[texIdx];
		if (!tex) return null;

		const transform = texInfo.extensions?.KHR_texture_transform;

		// ALWAYS clone here to avoid shared sampler settings between textures
		const cloned = tex.clone();
		if (colorSpace) cloned.colorSpace = colorSpace;

		if (transform) {
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
		}
		return cloned;
	}

	private _getTexCoord(texInfo: any): number {
		if (!texInfo) return 0;
		let uv = texInfo.texCoord ?? 0;
		if (texInfo.extensions?.KHR_texture_transform?.texCoord !== undefined) {
			uv = texInfo.extensions.KHR_texture_transform.texCoord;
		}
		return uv;
	}

	public parseMaterials(
		json: any,
		textures: (Texture | null)[] = []
	): Material[] {
		if (!json.materials) return [];
		return json.materials.map((m: any) => {
			const pbr = m.pbrMetallicRoughness || {};
			const baseColor = pbr.baseColorFactor || [1, 1, 1, 1];

			if (m.extensions?.KHR_materials_unlit) {
				const unlitMat = new UnlitMaterial({
					diffuse: {
						r: baseColor[0] * 255,
						g: baseColor[1] * 255,
						b: baseColor[2] * 255,
					},
					opacity: baseColor[3],
					doubleSided: m.doubleSided || false,
				});
				if (pbr.baseColorTexture !== undefined) {
					const tex = this._getMaterialTexture(
						pbr.baseColorTexture,
						textures,
						"sRGB"
					);
					if (tex) unlitMat.map = tex;
				}
				if (m.alphaMode !== undefined)
					(unlitMat as any).alphaMode = m.alphaMode;
				if (m.alphaCutoff !== undefined)
					(unlitMat as any).alphaCutoff = m.alphaCutoff;
				return unlitMat;
			}

			const material = new PBRMaterial({
				albedo: {
					r: baseColor[0] * 255,
					g: baseColor[1] * 255,
					b: baseColor[2] * 255,
				},
				opacity: baseColor[3],
				roughness:
					pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 1.0,
				metalness: pbr.metallicFactor !== undefined ? pbr.metallicFactor : 1.0,
				emissive: m.emissiveFactor
					? {
							r: m.emissiveFactor[0] * 255,
							g: m.emissiveFactor[1] * 255,
							b: m.emissiveFactor[2] * 255,
						}
					: { r: 0, g: 0, b: 0 },
				doubleSided: m.doubleSided || false,
			});
			if (pbr.baseColorTexture !== undefined) {
				const tex = this._getMaterialTexture(
					pbr.baseColorTexture,
					textures,
					"sRGB"
				);
				if (tex) {
					material.map = tex;
					material.albedoMapUV = this._getTexCoord(pbr.baseColorTexture);
				}
			}
			if (pbr.metallicRoughnessTexture !== undefined) {
				const tex = this._getMaterialTexture(
					pbr.metallicRoughnessTexture,
					textures,
					"Linear"
				);
				if (tex) {
					material.metallicRoughnessMap = tex;
					material.metallicRoughnessMapUV = this._getTexCoord(
						pbr.metallicRoughnessTexture
					);
				}
			}
			if (m.normalTexture !== undefined) {
				const tex = this._getMaterialTexture(
					m.normalTexture,
					textures,
					"Linear"
				);
				if (tex) {
					material.normalMap = tex;
					material.normalMapUV = this._getTexCoord(m.normalTexture);
					if (m.normalTexture.scale !== undefined) {
						material.normalScale = m.normalTexture.scale;
					}
				}
			}
			if (m.emissiveTexture !== undefined) {
				const tex = this._getMaterialTexture(
					m.emissiveTexture,
					textures,
					"sRGB"
				);
				if (tex) {
					material.emissiveMap = tex;
					material.emissiveMapUV = this._getTexCoord(m.emissiveTexture);
				}
			}
			if (m.occlusionTexture !== undefined) {
				const tex = this._getMaterialTexture(
					m.occlusionTexture,
					textures,
					"Linear"
				);
				if (tex) {
					material.occlusionMap = tex;
					material.occlusionMapUV = this._getTexCoord(m.occlusionTexture);
					if (m.occlusionTexture.strength !== undefined) {
						material.occlusionStrength = m.occlusionTexture.strength;
					}
				}
			}
			if (m.alphaMode !== undefined) (material as any).alphaMode = m.alphaMode;
			if (m.alphaCutoff !== undefined)
				(material as any).alphaCutoff = m.alphaCutoff;
			// KHR_materials_emissive_strength extension
			if (m.extensions?.KHR_materials_emissive_strength) {
				material.emissiveIntensity =
					m.extensions.KHR_materials_emissive_strength.emissiveStrength ?? 1.0;
			}
			// KHR_materials_ior extension
			if (m.extensions?.KHR_materials_ior) {
				material.ior = m.extensions.KHR_materials_ior.ior ?? 1.5;
			}
			// KHR_materials_specular extension
			if (m.extensions?.KHR_materials_specular) {
				const specExt = m.extensions.KHR_materials_specular;
				if (specExt.specularFactor !== undefined) {
					material.specularFactor = specExt.specularFactor;
				}
				if (specExt.specularColorFactor !== undefined) {
					const specColorFactor = specExt.specularColorFactor;
					material.specularColor = {
						r: (specColorFactor[0] ?? 1.0) * 255,
						g: (specColorFactor[1] ?? 1.0) * 255,
						b: (specColorFactor[2] ?? 1.0) * 255,
					};
				}
				if (specExt.specularTexture !== undefined) {
					const tex = this._getMaterialTexture(
						specExt.specularTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.specularMap = tex;
						material.specularMapUV = this._getTexCoord(specExt.specularTexture);
					}
				}
				if (specExt.specularColorTexture !== undefined) {
					const tex = this._getMaterialTexture(
						specExt.specularColorTexture,
						textures,
						"sRGB"
					);
					if (tex) {
						material.specularColorMap = tex;
						material.specularColorMapUV = this._getTexCoord(
							specExt.specularColorTexture
						);
					}
				}
			}
			// KHR_materials_clearcoat extension
			if (m.extensions?.KHR_materials_clearcoat) {
				const clearExt = m.extensions.KHR_materials_clearcoat;
				if (clearExt.clearcoatFactor !== undefined) {
					material.clearcoat = clearExt.clearcoatFactor;
				}
				if (clearExt.clearcoatRoughnessFactor !== undefined) {
					material.clearcoatRoughness = clearExt.clearcoatRoughnessFactor;
				}
				if (clearExt.clearcoatTexture !== undefined) {
					const tex = this._getMaterialTexture(
						clearExt.clearcoatTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.clearcoatMap = tex;
						material.clearcoatMapUV = this._getTexCoord(
							clearExt.clearcoatTexture
						);
					}
				}
				if (clearExt.clearcoatRoughnessTexture !== undefined) {
					const tex = this._getMaterialTexture(
						clearExt.clearcoatRoughnessTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.clearcoatRoughnessMap = tex;
						material.clearcoatRoughnessMapUV = this._getTexCoord(
							clearExt.clearcoatRoughnessTexture
						);
					}
				}
				if (clearExt.clearcoatNormalTexture !== undefined) {
					const tex = this._getMaterialTexture(
						clearExt.clearcoatNormalTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.clearcoatNormalMap = tex;
						material.clearcoatNormalMapUV = this._getTexCoord(
							clearExt.clearcoatNormalTexture
						);
						if (clearExt.clearcoatNormalTexture.scale !== undefined) {
							material.clearcoatNormalScale =
								clearExt.clearcoatNormalTexture.scale;
						}
					}
				}
			}
			// KHR_materials_sheen extension
			if (m.extensions?.KHR_materials_sheen) {
				const sheenExt = m.extensions.KHR_materials_sheen;
				if (sheenExt.sheenColorFactor !== undefined) {
					const f = sheenExt.sheenColorFactor;
					material.sheenColorFactor = {
						r: (f[0] ?? 0) * 255,
						g: (f[1] ?? 0) * 255,
						b: (f[2] ?? 0) * 255,
					};
				}
				if (sheenExt.sheenRoughnessFactor !== undefined) {
					material.sheenRoughnessFactor = sheenExt.sheenRoughnessFactor;
				}
				if (sheenExt.sheenColorTexture !== undefined) {
					const tex = this._getMaterialTexture(
						sheenExt.sheenColorTexture,
						textures,
						"sRGB"
					);
					if (tex) {
						material.sheenColorMap = tex;
						material.sheenColorMapUV = this._getTexCoord(
							sheenExt.sheenColorTexture
						);
					}
				}
				if (sheenExt.sheenRoughnessTexture !== undefined) {
					const tex = this._getMaterialTexture(
						sheenExt.sheenRoughnessTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.sheenRoughnessMap = tex;
						material.sheenRoughnessMapUV = this._getTexCoord(
							sheenExt.sheenRoughnessTexture
						);
					}
				}
			}
			// KHR_materials_transmission extension
			if (m.extensions?.KHR_materials_transmission) {
				const transExt = m.extensions.KHR_materials_transmission;
				if (transExt.transmissionFactor !== undefined) {
					material.transmissionFactor = transExt.transmissionFactor;
				}
				if (transExt.transmissionTexture !== undefined) {
					const tex = this._getMaterialTexture(
						transExt.transmissionTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.transmissionMap = tex;
						material.transmissionMapUV = this._getTexCoord(
							transExt.transmissionTexture
						);
					}
				}
			}
			// KHR_materials_volume extension
			if (m.extensions?.KHR_materials_volume) {
				const volExt = m.extensions.KHR_materials_volume;
				if (volExt.thicknessFactor !== undefined) {
					material.thicknessFactor = volExt.thicknessFactor;
				}
				if (volExt.thicknessTexture !== undefined) {
					const tex = this._getMaterialTexture(
						volExt.thicknessTexture,
						textures,
						"Linear"
					);
					if (tex) {
						material.thicknessMap = tex;
						material.thicknessMapUV = this._getTexCoord(
							volExt.thicknessTexture
						);
					}
				}
				if (volExt.attenuationDistance !== undefined) {
					material.attenuationDistance = volExt.attenuationDistance;
				}
				if (volExt.attenuationColor !== undefined) {
					const f = volExt.attenuationColor;
					material.attenuationColor = {
						r: (f[0] ?? 1) * 255,
						g: (f[1] ?? 1) * 255,
						b: (f[2] ?? 1) * 255,
					};
				}
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
						img.uri.startsWith("data:") || img.uri.startsWith("http")
							? img.uri
							: baseURL + img.uri;
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
			if (texture) {
				// Clone to avoid sharing sampler/transform state between textures
				const tex = texture.clone();
				if (t.sampler !== undefined) {
					const sampler = json.samplers[t.sampler];
					if (sampler.magFilter === 9728) tex.magFilter = "Nearest";
					else if (sampler.magFilter === 9729) tex.magFilter = "Linear";
					const minFilters: Record<number, string> = {
						9728: "Nearest",
						9729: "Linear",
						9984: "NearestMipmapNearest",
						9985: "LinearMipmapNearest",
						9986: "NearestMipmapLinear",
						9987: "LinearMipmapLinear",
					};
					if (sampler.minFilter !== undefined)
						tex.minFilter = minFilters[sampler.minFilter] || "Linear";
					const wrapModes: Record<
						number,
						"Repeat" | "Clamp" | "MirroredRepeat"
					> = {
						33071: "Clamp",
						10497: "Repeat",
						33648: "MirroredRepeat",
					};
					if (sampler.wrapS !== undefined)
						tex.wrapS = wrapModes[sampler.wrapS] || "Repeat";
					if (sampler.wrapT !== undefined)
						tex.wrapT = wrapModes[sampler.wrapT] || "Repeat";
				}
				return tex;
			}
			return null;
		});
	}

	public parseNodeTree(
		json: any,
		nodeIdx: number,
		buffers: Uint8Array[],
		materials: Material[],
		lights: any[],
		context: GLTFParseContext,
		parentPath: string
	): Node {
		if (nodeIdx === undefined || !json.nodes || !json.nodes[nodeIdx]) {
			return new Node({
				name: "emptyNode",
			});
		}
		const nodeDef = json.nodes[nodeIdx];
		const nodeName = nodeDef.name ?? `node_${nodeIdx}`;
		const nodePath = `${parentPath}/${sanitizePathSegment(nodeName)}_${nodeIdx}`;
		const container = new Node({
			name: nodeName,
		});
		this._applyNodeTransform(nodeDef, container);
		context.nodeByIndex.set(nodeIdx, container);
		context.nodePathByIndex.set(nodeIdx, nodePath);
		context.pathToNode.set(nodePath, container);

		if (
			nodeDef.mesh !== undefined &&
			json.meshes &&
			json.meshes[nodeDef.mesh]
		) {
			const mesh = this.parseMesh(
				json,
				nodeDef.mesh,
				buffers,
				materials,
				context
			);
			if (mesh) {
				const meshInstance = new MeshInstance({
					mesh,
					name: json.meshes[nodeDef.mesh]?.name ?? `mesh_${nodeDef.mesh}`,
					morphWeights:
						nodeDef.weights !== undefined
							? mesh.defaultMorphWeights.map((weights) =>
									applyMorphWeightOverride(weights, nodeDef.weights)
								)
							: undefined,
				});
				container.addChild(meshInstance);
				context.meshInstanceByNodeIndex.set(nodeIdx, meshInstance);
				context.pathToMeshInstance.set(nodePath, meshInstance);
				if (nodeDef.skin !== undefined) {
					context.pendingSkinByInstance.set(meshInstance, nodeDef.skin);
				}
			}
		}

		if (nodeDef.camera !== undefined && json.cameras?.[nodeDef.camera]) {
			const camera = this.parseCamera(json.cameras[nodeDef.camera]);
			container.addChild(camera);
		}

		const light = this.parseNodeLight(nodeDef, lights);
		if (light) {
			container.addChild(light);
		}

		if (nodeDef.children) {
			for (const childIdx of nodeDef.children) {
				container.addChild(
					this.parseNodeTree(
						json,
						childIdx,
						buffers,
						materials,
						lights,
						context,
						nodePath
					)
				);
			}
		}

		return container;
	}

	public parseMesh(
		json: any,
		meshIndex: number,
		buffers: Uint8Array[],
		materials: Material[],
		context: GLTFParseContext
	): MeshAsset | null {
		if (context.meshCache.has(meshIndex)) {
			return context.meshCache.get(meshIndex) ?? null;
		}

		const meshDef = json.meshes?.[meshIndex];
		if (!meshDef?.primitives || meshDef.primitives.length === 0) {
			context.meshCache.set(meshIndex, null);
			return null;
		}
		const primitives: IPrimitive[] = [];
		const defaultMorphWeights: Float32Array[] = [];
		for (const primitiveDef of meshDef.primitives) {
			const primitive = this.parsePrimitive(
				json,
				primitiveDef,
				buffers,
				materials
			);
			if (!primitive) continue;
			primitives.push(primitive);
			const targetCount = primitive.geometry.morphTargets?.length ?? 0;
			defaultMorphWeights.push(
				resolveDefaultMorphWeights(meshDef.weights, targetCount)
			);
		}
		if (primitives.length === 0) {
			context.meshCache.set(meshIndex, null);
			return null;
		}

		const mesh = new MeshAsset(primitives, defaultMorphWeights);
		context.meshCache.set(meshIndex, mesh);
		return mesh;
	}

	public parsePrimitive(
		json: any,
		primitive: any,
		buffers: Uint8Array[],
		materials: Material[]
	): IPrimitive | null {
		const attrs = primitive.attributes;
		const material =
			primitive.material !== undefined && materials[primitive.material]
				? materials[primitive.material]
				: new PBRMaterial();
		if (attrs.POSITION === undefined) return null;

		const positions = toFloat32Array(
			this.getAccessorData(json, buffers, attrs.POSITION)
		);
		const vertexCount = (positions.length / 3) | 0;
		const normals =
			attrs.NORMAL !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.NORMAL))
				: null;
		const tangents =
			attrs.TANGENT !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.TANGENT))
				: null;
		const uv0 =
			attrs.TEXCOORD_0 !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.TEXCOORD_0))
				: null;
		const uv1 =
			attrs.TEXCOORD_1 !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.TEXCOORD_1))
				: null;
		const colors =
			attrs.COLOR_0 !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.COLOR_0))
				: null;
		const joints0 =
			attrs.JOINTS_0 !== undefined
				? toJointArray(this.getAccessorData(json, buffers, attrs.JOINTS_0))
				: null;
		const weights0 =
			attrs.WEIGHTS_0 !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.WEIGHTS_0))
				: null;
		const joints1 =
			attrs.JOINTS_1 !== undefined
				? toJointArray(this.getAccessorData(json, buffers, attrs.JOINTS_1))
				: null;
		const weights1 =
			attrs.WEIGHTS_1 !== undefined
				? toFloat32Array(this.getAccessorData(json, buffers, attrs.WEIGHTS_1))
				: null;

		const sourceIndices =
			primitive.indices !== undefined
				? toUint32Array(this.getAccessorData(json, buffers, primitive.indices))
				: createSequentialIndices(vertexCount);
		const mode = primitive.mode;
		const topology = resolvePrimitiveTopology(mode);
		const indices = convertPrimitiveIndices(mode, sourceIndices);

		const morphTargets = this.parseMorphTargets(
			json,
			primitive.targets,
			buffers
		);

		const geometry: IPrimitiveGeometry = {
			positions,
			normals,
			tangents,
			uv0,
			uv1,
			colors,
			joints0,
			weights0,
			joints1,
			weights1,
			morphTargets,
			indices,
		};

		const boundingBox = GeometryBuilder.computeBoundingBox(geometry);
		const boundingSphere = GeometryBuilder.computeBoundingSphere(
			geometry,
			boundingBox
		);

		return {
			id: IdGenerator.nextId("primitive"),
			geometry,
			topology,
			material,
			boundingSphere,
			boundingBox,
			visible: true,
			castShadows: true,
			receiveShadows: true,
		};
	}

	public parseCamera(cameraDef: any): Camera {
		if (cameraDef.type === "orthographic") {
			const xmag = cameraDef.orthographic?.xmag ?? 1;
			const ymag = cameraDef.orthographic?.ymag ?? 1;
			const camera = new OrthographicCamera(ymag * 2);
			camera.setBounds(-xmag, xmag, -ymag, ymag);
			camera.type = CameraType.Orthographic;
			camera.near = cameraDef.orthographic?.znear ?? camera.near;
			camera.far = cameraDef.orthographic?.zfar ?? camera.far;
			camera.updateMatrices();
			return camera;
		}

		const camera = new Camera();
		camera.type = CameraType.Perspective;
		const perspective = cameraDef.perspective ?? {};
		camera.fov = ((perspective.yfov ?? Math.PI / 3) * 180) / Math.PI;
		if (perspective.aspectRatio !== undefined) {
			camera.aspectRatio = perspective.aspectRatio;
		}
		camera.near = perspective.znear ?? camera.near;
		camera.far = perspective.zfar ?? camera.far;
		camera.updateMatrices();
		return camera;
	}

	public parseNodeLight(nodeDef: any, lights: any[]): SceneLight | null {
		const lightIndex = nodeDef.extensions?.KHR_lights_punctual?.light;
		if (lightIndex === undefined) return null;
		const lightDef = lights?.[lightIndex];
		if (!lightDef) return null;

		const colorFactor = lightDef.color ?? [1, 1, 1];
		const color = {
			r: (colorFactor[0] ?? 1) * 255,
			g: (colorFactor[1] ?? 1) * 255,
			b: (colorFactor[2] ?? 1) * 255,
		};
		const intensity = lightDef.intensity ?? 1;
		switch (lightDef.type) {
			case "directional":
				return new DirectionalLight({
					color,
					intensity,
					direction: { x: 0, y: 0, z: -1 },
				});
			case "point":
				return new PointLight({
					color,
					intensity,
					range: lightDef.range ?? 1000,
				});
			case "spot":
				return new SpotLight({
					color,
					intensity,
					range: lightDef.range ?? 1000,
					direction: { x: 0, y: 0, z: -1 },
					innerAngle: lightDef.spot?.innerConeAngle ?? 0,
					angle: lightDef.spot?.outerConeAngle ?? Math.PI / 4,
				});
			default:
				return new AmbientLight({
					color,
					intensity,
				});
		}
	}

	public parseMorphTargets(
		json: any,
		targetDefs: any[] | undefined,
		buffers: Uint8Array[]
	): MorphTargetGeometry[] | null {
		if (!Array.isArray(targetDefs) || targetDefs.length === 0) {
			return null;
		}

		return targetDefs.map((targetDef) => ({
			positions:
				targetDef.POSITION !== undefined
					? toFloat32Array(
							this.getAccessorData(json, buffers, targetDef.POSITION)
						)
					: null,
			normals:
				targetDef.NORMAL !== undefined
					? toFloat32Array(
							this.getAccessorData(json, buffers, targetDef.NORMAL)
						)
					: null,
			tangents:
				targetDef.TANGENT !== undefined
					? toFloat32Array(
							this.getAccessorData(json, buffers, targetDef.TANGENT)
						)
					: null,
		}));
	}

	public parseSkins(
		json: any,
		buffers: Uint8Array[],
		context: GLTFParseContext
	): Skeleton[] {
		const skins = json.skins ?? [];
		const result: Skeleton[] = [];
		for (let skinIndex = 0; skinIndex < skins.length; skinIndex++) {
			const skinDef = skins[skinIndex];
			if (!Array.isArray(skinDef?.joints) || skinDef.joints.length === 0) {
				continue;
			}

			const joints = skinDef.joints
				.map((jointIndex: number) => context.nodeByIndex.get(jointIndex))
				.filter(Boolean) as Node[];
			if (joints.length !== skinDef.joints.length) {
				console.warn(
					`GLTFLoader: skin ${skinIndex} has missing joint node bindings; skipping`
				);
				continue;
			}

			let inverseBindMatrices: Matrix4[] = [];
			if (skinDef.inverseBindMatrices !== undefined) {
				const data = toFloat32Array(
					this.getAccessorData(json, buffers, skinDef.inverseBindMatrices)
				);
				for (let i = 0; i < joints.length; i++) {
					const offset = i * 16;
					inverseBindMatrices.push(
						Matrix4.fromArray(Array.from(data.subarray(offset, offset + 16)))
					);
				}
			}

			if (inverseBindMatrices.length !== joints.length) {
				inverseBindMatrices = joints.map(() => Matrix4.identity());
			}

			const skeleton = new Skeleton({
				name: skinDef.name ?? `skin_${skinIndex}`,
				joints,
				inverseBindMatrices,
			});
			context.skeletonBySkinIndex.set(skinIndex, skeleton);
			result.push(skeleton);
		}

		for (const [
			instance,
			skinIndex,
		] of context.pendingSkinByInstance.entries()) {
			const skeleton = context.skeletonBySkinIndex.get(skinIndex);
			if (!skeleton) continue;
			instance.skeleton = skeleton;
		}

		return result;
	}

	public parseAnimations(
		json: any,
		buffers: Uint8Array[],
		context: GLTFParseContext
	): AnimationClip[] {
		const animations = json.animations ?? [];
		const clips: AnimationClip[] = [];

		for (
			let animationIndex = 0;
			animationIndex < animations.length;
			animationIndex++
		) {
			const animationDef = animations[animationIndex];
			const tracks: KeyframeTrack[] = [];
			let duration = 0;

			for (const channel of animationDef.channels ?? []) {
				const sampler = animationDef.samplers?.[channel.sampler];
				if (!sampler) continue;
				const nodeIndex = channel.target?.node;
				const targetPath = channel.target?.path;
				if (nodeIndex === undefined || !targetPath) continue;

				const nodePath = context.nodePathByIndex.get(nodeIndex);
				if (!nodePath) continue;

				const inputTimes = toFloat32Array(
					this.getAccessorData(json, buffers, sampler.input)
				);
				if (inputTimes.length === 0) continue;
				duration = Math.max(duration, inputTimes[inputTimes.length - 1] ?? 0);

				const outputValues = toFloat32Array(
					this.getAccessorData(json, buffers, sampler.output)
				);
				const interpolation = mapInterpolation(sampler.interpolation);
				let binding: any = null;
				let valueSize = 0;

				if (
					targetPath === "translation" ||
					targetPath === "rotation" ||
					targetPath === "scale"
				) {
					binding = {
						targetType: "node",
						targetPath: nodePath,
						property: targetPath,
					};
					valueSize = targetPath === "rotation" ? 4 : 3;
				} else if (targetPath === "weights") {
					if (!context.pathToMeshInstance.has(nodePath)) {
						console.warn(
							`GLTFLoader: animation channel targets weights on node ${nodePath} without mesh; ignored`
						);
						continue;
					}
					binding = {
						targetType: "morph",
						targetPath: nodePath,
						property: "weights",
					};
					valueSize =
						interpolation === "cubic"
							? Math.floor(outputValues.length / (inputTimes.length * 3))
							: Math.floor(outputValues.length / inputTimes.length);
				} else {
					console.warn(
						`GLTFLoader: unsupported animation path "${targetPath}" ignored`
					);
					continue;
				}

				if (!Number.isFinite(valueSize) || valueSize <= 0) continue;

				tracks.push(
					new KeyframeTrack({
						name: `${animationDef.name ?? `animation_${animationIndex}`}:${targetPath}`,
						binding,
						times: inputTimes,
						values: outputValues,
						valueSize,
						interpolation,
					})
				);
			}

			if (tracks.length === 0) continue;
			clips.push(
				new AnimationClip({
					name: animationDef.name ?? `animation_${animationIndex}`,
					duration,
					tracks,
				})
			);
		}

		return clips;
	}

	private _applyNodeTransform(nodeDef: any, target: Node): void {
		if (nodeDef.matrix) {
			const matrix = Matrix4.fromArray(nodeDef.matrix);
			const decomposed = decomposeTRS(matrix);
			target.position.copy(decomposed.position);
			target.quaternion = decomposed.quaternion;
			target.scale.copy(decomposed.scale);
			target.updateLocalMatrix();
			return;
		}

		if (nodeDef.translation) {
			target.position.set(
				nodeDef.translation[0] ?? 0,
				nodeDef.translation[1] ?? 0,
				nodeDef.translation[2] ?? 0
			);
		}

		if (nodeDef.rotation) {
			target.quaternion = new Quaternion(
				nodeDef.rotation[0] ?? 0,
				nodeDef.rotation[1] ?? 0,
				nodeDef.rotation[2] ?? 0,
				nodeDef.rotation[3] ?? 1
			).normalize();
		}

		if (nodeDef.scale) {
			target.scale.set(
				nodeDef.scale[0] ?? 1,
				nodeDef.scale[1] ?? 1,
				nodeDef.scale[2] ?? 1
			);
		}

		target.updateLocalMatrix();
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
				[TYPE_MAT2]: 4,
				[TYPE_MAT3]: 9,
				[TYPE_MAT4]: 16,
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

function decomposeTRS(matrix: Matrix4): {
	position: { x: number; y: number; z: number };
	quaternion: Quaternion;
	scale: { x: number; y: number; z: number };
} {
	const m = matrix.elements;
	const position = {
		x: m[0][3],
		y: m[1][3],
		z: m[2][3],
	};
	const scale = {
		x: Math.hypot(m[0][0], m[1][0], m[2][0]) || 1,
		y: Math.hypot(m[0][1], m[1][1], m[2][1]) || 1,
		z: Math.hypot(m[0][2], m[1][2], m[2][2]) || 1,
	};
	const rotationMatrix = [
		[m[0][0] / scale.x, m[0][1] / scale.y, m[0][2] / scale.z],
		[m[1][0] / scale.x, m[1][1] / scale.y, m[1][2] / scale.z],
		[m[2][0] / scale.x, m[2][1] / scale.y, m[2][2] / scale.z],
	];

	return {
		position,
		quaternion: Quaternion.fromRotationMatrix(rotationMatrix),
		scale,
	};
}

function sanitizePathSegment(value: string): string {
	return value.replace(/[^\w\-]+/g, "_");
}

function resolveDefaultMorphWeights(
	meshWeights: ArrayLike<number> | undefined,
	targetCount: number
): Float32Array {
	const result = new Float32Array(targetCount);
	for (let i = 0; i < targetCount; i++) {
		result[i] = Number(meshWeights?.[i] ?? 0);
	}
	return result;
}

function applyMorphWeightOverride(
	baseWeights: Float32Array,
	overrideWeights: ArrayLike<number>
): Float32Array {
	const result = new Float32Array(baseWeights);
	const count = Math.min(result.length, overrideWeights.length);
	for (let i = 0; i < count; i++) {
		result[i] = Number(overrideWeights[i] ?? result[i]);
	}
	return result;
}

function mapInterpolation(value: string | undefined) {
	switch ((value ?? "LINEAR").toUpperCase()) {
		case "STEP":
			return "step" as const;
		case "CUBICSPLINE":
			return "cubic" as const;
		case "LINEAR":
		default:
			return "linear" as const;
	}
}

function toFloat32Array(
	value:
		| Float32Array
		| Uint32Array
		| Uint16Array
		| Uint8Array
		| Int16Array
		| Int8Array
		| number[]
): Float32Array {
	if (value instanceof Float32Array) return value;
	return new Float32Array(value as ArrayLike<number>);
}

function toUint32Array(
	value:
		| Uint32Array
		| Uint16Array
		| Uint8Array
		| Int16Array
		| Int8Array
		| Float32Array
		| number[]
): Uint32Array {
	if (value instanceof Uint32Array) return value;
	return new Uint32Array(value as ArrayLike<number>);
}

function toJointArray(
	value:
		| Uint16Array
		| Uint32Array
		| Uint8Array
		| Int16Array
		| Int8Array
		| Float32Array
		| number[]
): Uint16Array | Uint32Array {
	if (value instanceof Uint32Array || value instanceof Uint16Array) {
		return value;
	}
	return new Uint16Array(value as ArrayLike<number>);
}

function createSequentialIndices(vertexCount: number): Uint32Array {
	const indices = new Uint32Array(vertexCount);
	for (let i = 0; i < vertexCount; i++) {
		indices[i] = i;
	}
	return indices;
}

function resolvePrimitiveTopology(mode: number | undefined): PrimitiveDrawTopology {
	switch (mode ?? GLTF_MODE_TRIANGLES) {
		case GLTF_MODE_POINTS:
			return "point-list";
		case GLTF_MODE_LINES:
		case GLTF_MODE_LINE_LOOP:
		case GLTF_MODE_LINE_STRIP:
			return "line-list";
		case GLTF_MODE_TRIANGLES:
		case GLTF_MODE_TRIANGLE_STRIP:
		case GLTF_MODE_TRIANGLE_FAN:
			return DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		default:
			throw new Error(`Unsupported glTF primitive mode: ${mode}`);
	}
}

function convertPrimitiveIndices(
	mode: number | undefined,
	sourceIndices: Uint32Array
): Uint32Array {
	const resolvedMode = mode ?? GLTF_MODE_TRIANGLES;
	switch (resolvedMode) {
		case GLTF_MODE_POINTS:
			return sourceIndices;
		case GLTF_MODE_LINES:
			return normalizeLineListIndices(sourceIndices);
		case GLTF_MODE_LINE_LOOP:
			return convertLineLoopIndices(sourceIndices);
		case GLTF_MODE_LINE_STRIP:
			return convertLineStripIndices(sourceIndices);
		case GLTF_MODE_TRIANGLES:
			return normalizeTriangleListIndices(sourceIndices);
		case GLTF_MODE_TRIANGLE_STRIP:
			return convertTriangleStripIndices(sourceIndices);
		case GLTF_MODE_TRIANGLE_FAN:
			return convertTriangleFanIndices(sourceIndices);
		default:
			throw new Error(`Unsupported glTF primitive mode: ${mode}`);
	}
}

function normalizeLineListIndices(sourceIndices: Uint32Array): Uint32Array {
	const pairCount = Math.floor(sourceIndices.length / 2);
	if (pairCount * 2 === sourceIndices.length) {
		return sourceIndices;
	}
	return sourceIndices.slice(0, pairCount * 2);
}

function convertLineLoopIndices(sourceIndices: Uint32Array): Uint32Array {
	const pointCount = sourceIndices.length;
	if (pointCount < 2) return new Uint32Array(0);

	const indices = new Uint32Array(pointCount * 2);
	let cursor = 0;
	for (let i = 0; i < pointCount - 1; i++) {
		indices[cursor++] = sourceIndices[i];
		indices[cursor++] = sourceIndices[i + 1];
	}
	indices[cursor++] = sourceIndices[pointCount - 1];
	indices[cursor++] = sourceIndices[0];
	return indices;
}

function convertLineStripIndices(sourceIndices: Uint32Array): Uint32Array {
	const pointCount = sourceIndices.length;
	if (pointCount < 2) return new Uint32Array(0);

	const indices = new Uint32Array((pointCount - 1) * 2);
	let cursor = 0;
	for (let i = 0; i < pointCount - 1; i++) {
		indices[cursor++] = sourceIndices[i];
		indices[cursor++] = sourceIndices[i + 1];
	}
	return indices;
}

function normalizeTriangleListIndices(sourceIndices: Uint32Array): Uint32Array {
	const triangleCount = Math.floor(sourceIndices.length / 3);
	if (triangleCount * 3 === sourceIndices.length) {
		return sourceIndices;
	}
	return sourceIndices.slice(0, triangleCount * 3);
}

function convertTriangleStripIndices(sourceIndices: Uint32Array): Uint32Array {
	const pointCount = sourceIndices.length;
	if (pointCount < 3) return new Uint32Array(0);

	const triangleCount = pointCount - 2;
	const indices = new Uint32Array(triangleCount * 3);
	let cursor = 0;
	for (let i = 0; i < triangleCount; i++) {
		if ((i & 1) === 0) {
			indices[cursor++] = sourceIndices[i];
			indices[cursor++] = sourceIndices[i + 1];
			indices[cursor++] = sourceIndices[i + 2];
		} else {
			indices[cursor++] = sourceIndices[i + 1];
			indices[cursor++] = sourceIndices[i];
			indices[cursor++] = sourceIndices[i + 2];
		}
	}
	return indices;
}

function convertTriangleFanIndices(sourceIndices: Uint32Array): Uint32Array {
	const pointCount = sourceIndices.length;
	if (pointCount < 3) return new Uint32Array(0);

	const triangleCount = pointCount - 2;
	const center = sourceIndices[0];
	const indices = new Uint32Array(triangleCount * 3);
	let cursor = 0;
	for (let i = 1; i < pointCount - 1; i++) {
		indices[cursor++] = center;
		indices[cursor++] = sourceIndices[i];
		indices[cursor++] = sourceIndices[i + 1];
	}
	return indices;
}
