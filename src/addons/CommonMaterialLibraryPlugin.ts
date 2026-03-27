import { Texture } from "../core/Texture";
import { PBRMaterial, type PBRMaterialParams } from "../materials/PBRMaterial";

const DEFAULT_NORMAL_MAP_SIZE = 64;
const MIN_NORMAL_MAP_SIZE = 8;
const MAX_NORMAL_MAP_SIZE = 512;
const TAU = Math.PI * 2;

export type CommonMaterialPresetCategory =
	| "liquid"
	| "metal"
	| "polymer"
	| "glass"
	| "ceramic"
	| "emissive"
	| "surface";

export type CommonMetalPresetStyle =
	| "brushed-steel"
	| "polished-steel"
	| "gold"
	| "chrome"
	| "copper";

export type CommonPlasticPresetStyle =
	| "matte-plastic"
	| "glossy-plastic"
	| "rubber";

export interface CommonMaterialLibraryPluginOptions {
	normalMapSize?: number;
}

export interface CommonMaterialPresetFactoryContext {
	normalMapSize: number;
}

export type CommonMaterialPresetFactory = (
	context: CommonMaterialPresetFactoryContext
) => PBRMaterialParams;

export interface CommonMaterialPresetInfo {
	id: string;
	name: string;
	description: string;
	category: CommonMaterialPresetCategory | string;
	tags: string[];
}

export interface CommonMaterialPresetDefinition extends CommonMaterialPresetInfo {
	factory: CommonMaterialPresetFactory;
}

interface InternalPreset extends CommonMaterialPresetDefinition {
	builtIn: boolean;
}

/**
 * Preset material library plugin.
 *
 * Provides common physically based material presets and registration hooks so
 * users can quickly instantiate reusable material variants.
 */
export class CommonMaterialLibraryPlugin {
	private _normalMapSize = DEFAULT_NORMAL_MAP_SIZE;
	private _presets = new Map<string, InternalPreset>();

	constructor(options: CommonMaterialLibraryPluginOptions = {}) {
		this._normalMapSize = sanitizeTextureSize(
			options.normalMapSize ?? DEFAULT_NORMAL_MAP_SIZE
		);
		this._registerBuiltInPresets();
	}

	/**
	 * Resolution used when generating simulated normal map textures.
	 */
	public get normalMapSize(): number {
		return this._normalMapSize;
	}

	/**
	 * Updates the simulated normal map resolution.
	 */
	public setNormalMapSize(size: number): this {
		this._normalMapSize = sanitizeTextureSize(size);
		return this;
	}

	/**
	 * Lists all registered presets in deterministic id order.
	 */
	public listPresets(): CommonMaterialPresetInfo[] {
		return [...this._presets.values()]
			.sort((a, b) => a.id.localeCompare(b.id))
			.map((preset) => ({
				id: preset.id,
				name: preset.name,
				description: preset.description,
				category: preset.category,
				tags: [...preset.tags],
			}));
	}

	/**
	 * Lists all registered preset ids.
	 */
	public listPresetIds(): string[] {
		return this.listPresets().map((preset) => preset.id);
	}

	/**
	 * Returns whether a preset id exists.
	 */
	public hasPreset(id: string): boolean {
		return this._presets.has(normalizePresetId(id));
	}

	/**
	 * Returns metadata of a preset id.
	 */
	public getPreset(id: string): CommonMaterialPresetInfo | null {
		const preset = this._presets.get(normalizePresetId(id));
		if (!preset) return null;
		return {
			id: preset.id,
			name: preset.name,
			description: preset.description,
			category: preset.category,
			tags: [...preset.tags],
		};
	}

	/**
	 * Registers a custom material preset.
	 */
	public registerPreset(definition: CommonMaterialPresetDefinition): this {
		this._registerInternalPreset(definition, false);
		return this;
	}

	/**
	 * Removes a custom material preset.
	 *
	 * Returns false when the preset does not exist or is built-in.
	 */
	public unregisterPreset(id: string): boolean {
		const key = normalizePresetId(id);
		const preset = this._presets.get(key);
		if (!preset || preset.builtIn) return false;
		return this._presets.delete(key);
	}

	/**
	 * Instantiates a material from preset id and optional overrides.
	 */
	public createMaterial(
		id: string,
		overrides: PBRMaterialParams = {}
	): PBRMaterial {
		const preset = this._resolvePreset(id);
		const defaults = preset.factory({
			normalMapSize: this._normalMapSize,
		});
		const merged = mergeParams(preset.name, defaults, overrides);
		return new PBRMaterial(merged);
	}

	/**
	 * Convenience helper for the default water preset.
	 */
	public createWaterMaterial(
		overrides: PBRMaterialParams = {}
	): PBRMaterial {
		return this.createMaterial("water", overrides);
	}

	/**
	 * Convenience helper for metallic presets.
	 */
	public createMetalMaterial(
		style: CommonMetalPresetStyle = "brushed-steel",
		overrides: PBRMaterialParams = {}
	): PBRMaterial {
		return this.createMaterial(style, overrides);
	}

	/**
	 * Convenience helper for polymer/plastic presets.
	 */
	public createPlasticMaterial(
		style: CommonPlasticPresetStyle = "matte-plastic",
		overrides: PBRMaterialParams = {}
	): PBRMaterial {
		return this.createMaterial(style, overrides);
	}

	/**
	 * Convenience helper for procedural normal-map simulation preset.
	 */
	public createSimulatedNormalMapMaterial(
		overrides: PBRMaterialParams = {}
	): PBRMaterial {
		return this.createMaterial("simulated-normal-map", overrides);
	}

	private _resolvePreset(id: string): InternalPreset {
		const key = normalizePresetId(id);
		const preset = this._presets.get(key);
		if (!preset) {
			throw new Error(
				`Unknown material preset "${id}". Registered presets: ${this.listPresetIds().join(", ")}`
			);
		}
		return preset;
	}

	private _registerInternalPreset(
		definition: CommonMaterialPresetDefinition | null | undefined,
		builtIn: boolean
	): void {
		if (!isRecord(definition)) {
			throw new Error("Material preset definition must be an object");
		}
		const id = normalizePresetId(
			typeof definition.id === "string" ? definition.id : ""
		);
		if (!id) {
			throw new Error("Material preset id must be a non-empty string");
		}
		if (this._presets.has(id)) {
			throw new Error(`Material preset "${id}" is already registered`);
		}
		const factory = definition.factory;
		if (typeof factory !== "function") {
			throw new Error(`Material preset "${id}" is missing a factory`);
		}
		const normalized: InternalPreset = {
			id,
			name: normalizeLabel(
				typeof definition.name === "string" ? definition.name : "",
				id
			),
			description: normalizeLabel(
				typeof definition.description === "string" ?
					definition.description
				:	"",
				""
			),
			category: normalizeCategory(definition.category),
			tags: normalizeTags(definition.tags),
			factory,
			builtIn,
		};
		this._presets.set(id, normalized);
	}

	private _registerBuiltInPresets(): void {
		this._registerInternalPreset(
			{
				id: "water",
				name: "Water",
				description: "Clear water surface with transmission and clearcoat.",
				category: "liquid",
				tags: ["water", "liquid", "transparent", "pbr"],
				factory: () => ({
					name: "Water",
					albedo: { r: 88, g: 148, b: 188 },
					metalness: 0.02,
					roughness: 0.08,
					ior: 1.333,
					transmissionFactor: 0.98,
					thicknessFactor: 0.5,
					attenuationDistance: 1.8,
					attenuationColor: { r: 192, g: 235, b: 255 },
					clearcoat: 1.0,
					clearcoatRoughness: 0.05,
					normalScale: 1.3,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "ocean-water",
				name: "Ocean Water",
				description: "Darker and rougher water suitable for sea surfaces.",
				category: "liquid",
				tags: ["water", "ocean", "liquid", "pbr"],
				factory: () => ({
					name: "Ocean Water",
					albedo: { r: 44, g: 92, b: 132 },
					metalness: 0.02,
					roughness: 0.18,
					ior: 1.333,
					transmissionFactor: 0.94,
					thicknessFactor: 0.85,
					attenuationDistance: 2.6,
					attenuationColor: { r: 140, g: 196, b: 224 },
					clearcoat: 1.0,
					clearcoatRoughness: 0.1,
					normalScale: 1.9,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "pool-water",
				name: "Pool Water",
				description: "Cleaner and calmer water for stylized pools.",
				category: "liquid",
				tags: ["water", "pool", "liquid", "pbr"],
				factory: () => ({
					name: "Pool Water",
					albedo: { r: 110, g: 196, b: 228 },
					metalness: 0.01,
					roughness: 0.05,
					ior: 1.333,
					transmissionFactor: 0.99,
					thicknessFactor: 0.35,
					attenuationDistance: 1.2,
					attenuationColor: { r: 210, g: 245, b: 255 },
					clearcoat: 0.95,
					clearcoatRoughness: 0.03,
					normalScale: 0.85,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "brushed-steel",
				name: "Brushed Steel",
				description: "General-purpose metal with anisotropic-like roughness feel.",
				category: "metal",
				tags: ["metal", "steel", "industrial", "pbr"],
				factory: () => ({
					name: "Brushed Steel",
					albedo: { r: 186, g: 190, b: 194 },
					metalness: 1.0,
					roughness: 0.34,
					reflectance: 1.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "polished-steel",
				name: "Polished Steel",
				description: "High-reflection polished metal surface.",
				category: "metal",
				tags: ["metal", "steel", "polished", "pbr"],
				factory: () => ({
					name: "Polished Steel",
					albedo: { r: 218, g: 224, b: 228 },
					metalness: 1.0,
					roughness: 0.09,
					reflectance: 1.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "gold",
				name: "Gold",
				description: "Bright metallic gold preset.",
				category: "metal",
				tags: ["metal", "gold", "jewelry", "pbr"],
				factory: () => ({
					name: "Gold",
					albedo: { r: 255, g: 214, b: 122 },
					metalness: 1.0,
					roughness: 0.18,
					reflectance: 1.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "chrome",
				name: "Chrome",
				description: "Mirror-like chrome metal.",
				category: "metal",
				tags: ["metal", "chrome", "mirror", "pbr"],
				factory: () => ({
					name: "Chrome",
					albedo: { r: 236, g: 240, b: 245 },
					metalness: 1.0,
					roughness: 0.03,
					reflectance: 1.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "copper",
				name: "Copper",
				description: "Warm copper metal material.",
				category: "metal",
				tags: ["metal", "copper", "warm", "pbr"],
				factory: () => ({
					name: "Copper",
					albedo: { r: 232, g: 158, b: 116 },
					metalness: 1.0,
					roughness: 0.24,
					reflectance: 1.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "matte-plastic",
				name: "Matte Plastic",
				description: "Diffused plastic suitable for consumer products.",
				category: "polymer",
				tags: ["plastic", "polymer", "matte", "pbr"],
				factory: () => ({
					name: "Matte Plastic",
					albedo: { r: 212, g: 216, b: 222 },
					metalness: 0.0,
					roughness: 0.62,
					ior: 1.46,
					clearcoat: 0.18,
					clearcoatRoughness: 0.32,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "glossy-plastic",
				name: "Glossy Plastic",
				description: "Hard glossy plastic with stronger highlights.",
				category: "polymer",
				tags: ["plastic", "polymer", "glossy", "pbr"],
				factory: () => ({
					name: "Glossy Plastic",
					albedo: { r: 230, g: 230, b: 234 },
					metalness: 0.0,
					roughness: 0.18,
					ior: 1.5,
					clearcoat: 0.86,
					clearcoatRoughness: 0.07,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "rubber",
				name: "Rubber",
				description: "Soft, low-reflective rubber surface.",
				category: "polymer",
				tags: ["rubber", "polymer", "matte", "pbr"],
				factory: () => ({
					name: "Rubber",
					albedo: { r: 58, g: 58, b: 62 },
					metalness: 0.0,
					roughness: 0.9,
					ior: 1.47,
					clearcoat: 0.0,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "ceramic",
				name: "Ceramic",
				description: "Glazed ceramic with mild clearcoat.",
				category: "ceramic",
				tags: ["ceramic", "glaze", "hard-surface", "pbr"],
				factory: () => ({
					name: "Ceramic",
					albedo: { r: 236, g: 236, b: 236 },
					metalness: 0.0,
					roughness: 0.28,
					ior: 1.52,
					clearcoat: 0.64,
					clearcoatRoughness: 0.08,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "frosted-glass",
				name: "Frosted Glass",
				description: "Semi-opaque frosted transmissive glass.",
				category: "glass",
				tags: ["glass", "frosted", "transmission", "pbr"],
				factory: () => ({
					name: "Frosted Glass",
					albedo: { r: 222, g: 232, b: 255 },
					metalness: 0.0,
					roughness: 0.42,
					ior: 1.52,
					transmissionFactor: 0.92,
					thicknessFactor: 0.12,
					attenuationDistance: 0.8,
					attenuationColor: { r: 226, g: 236, b: 255 },
					clearcoat: 0.2,
					clearcoatRoughness: 0.24,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "neon-emissive",
				name: "Neon Emissive",
				description: "Bright emissive material for signs and VFX accents.",
				category: "emissive",
				tags: ["emissive", "neon", "vfx", "pbr"],
				factory: () => ({
					name: "Neon Emissive",
					albedo: { r: 26, g: 26, b: 30 },
					metalness: 0.0,
					roughness: 0.22,
					ior: 1.44,
					emissive: { r: 96, g: 212, b: 255 },
					emissiveIntensity: 2.4,
					clearcoat: 0.35,
					clearcoatRoughness: 0.15,
				}),
			},
			true
		);
		this._registerInternalPreset(
			{
				id: "simulated-normal-map",
				name: "Simulated Normal Map",
				description:
					"Procedural normal map preset for quick high-frequency surface detail.",
				category: "surface",
				tags: ["normal-map", "procedural", "surface", "pbr"],
				factory: (context) => ({
					name: "Simulated Normal Map",
					albedo: { r: 188, g: 192, b: 198 },
					metalness: 0.04,
					roughness: 0.5,
					ior: 1.46,
					clearcoat: 0.32,
					clearcoatRoughness: 0.18,
					normalMap: createProceduralNormalMapTexture(context.normalMapSize),
					normalScale: 2.2,
				}),
			},
			true
		);
	}
}

function mergeParams(
	presetName: string,
	defaults: PBRMaterialParams,
	overrides: PBRMaterialParams
): PBRMaterialParams {
	const merged: PBRMaterialParams = {
		name: presetName,
		...defaults,
		...overrides,
	};
	if (typeof merged.name !== "string" || merged.name.trim().length === 0) {
		merged.name = presetName;
	}
	return merged;
}

function normalizePresetId(value: string): string {
	if (typeof value !== "string") return "";
	return value.trim().toLowerCase();
}

function normalizeLabel(value: string, fallback: string): string {
	if (typeof value !== "string") return fallback;
	const trimmed = value.trim();
	return trimmed.length > 0 ? trimmed : fallback;
}

function normalizeCategory(value: unknown): string {
	if (typeof value !== "string") return "surface";
	const trimmed = value.trim();
	return trimmed.length > 0 ? trimmed : "surface";
}

function normalizeTags(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value
		.filter((tag) => typeof tag === "string" && tag.trim().length > 0)
		.map((tag) => tag.trim());
}

function isRecord(
	value: CommonMaterialPresetDefinition | null | undefined
): value is CommonMaterialPresetDefinition {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sanitizeTextureSize(value: number): number {
	if (!Number.isFinite(value)) return DEFAULT_NORMAL_MAP_SIZE;
	const rounded = Math.round(Number(value));
	if (rounded < MIN_NORMAL_MAP_SIZE) return MIN_NORMAL_MAP_SIZE;
	if (rounded > MAX_NORMAL_MAP_SIZE) return MAX_NORMAL_MAP_SIZE;
	return rounded;
}

function createProceduralNormalMapTexture(size: number): Texture {
	const safeSize = sanitizeTextureSize(size);
	const data = new Uint8ClampedArray(safeSize * safeSize * 4);

	const du = 1 / safeSize;
	const dv = 1 / safeSize;
	for (let y = 0; y < safeSize; y++) {
		for (let x = 0; x < safeSize; x++) {
			const u = x / safeSize;
			const v = y / safeSize;

			const hL = sampleHeight(u - du, v);
			const hR = sampleHeight(u + du, v);
			const hD = sampleHeight(u, v - dv);
			const hU = sampleHeight(u, v + dv);

			let nx = (hL - hR) * 1.25;
			let ny = (hD - hU) * 1.25;
			let nz = 1.0;

			const invLength = 1 / Math.max(1e-8, Math.hypot(nx, ny, nz));
			nx *= invLength;
			ny *= invLength;
			nz *= invLength;

			const idx = (y * safeSize + x) * 4;
			data[idx] = Math.round((nx * 0.5 + 0.5) * 255);
			data[idx + 1] = Math.round((ny * 0.5 + 0.5) * 255);
			data[idx + 2] = Math.round((nz * 0.5 + 0.5) * 255);
			data[idx + 3] = 255;
		}
	}

	const texture = new Texture(data, safeSize, safeSize, "Linear");
	texture.wrapS = "Repeat";
	texture.wrapT = "Repeat";
	texture.minFilter = "Linear";
	texture.magFilter = "Linear";
	return texture;
}

function sampleHeight(u: number, v: number): number {
	const waveA = Math.sin((u * 12.4 + v * 5.6) * TAU);
	const waveB = Math.cos((u * -7.8 + v * 10.2) * TAU);
	const waveC = Math.sin((u + v) * 17.3 * TAU);
	return waveA * 0.55 + waveB * 0.3 + waveC * 0.15;
}
