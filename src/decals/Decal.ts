import type { Material } from "../materials/Material";
import { Node, normalizeRenderLayerMask, type NodeParams } from "../core/Node";

export const DECAL_BLEND_MODES = [
	"disabled",
	"lerp",
	"replace",
	"multiply",
	"add",
	"normal",
] as const;

export type DecalBlendMode = (typeof DECAL_BLEND_MODES)[number];

export const DECAL_CHANNELS = [
	"baseColor",
	"normal",
	"roughness",
	"metalness",
	"emissive",
	"occlusion",
	"specular",
	"specularColor",
	"clearcoat",
	"clearcoatRoughness",
	"clearcoatNormal",
	"sheenColor",
	"sheenRoughness",
	"transmission",
	"thickness",
	"iridescence",
	"iridescenceThickness",
	"anisotropy",
] as const;

export type DecalChannel = (typeof DECAL_CHANNELS)[number];
export type DecalChannelBlendModes = Partial<Record<DecalChannel, DecalBlendMode>>;

export const DEFAULT_DECAL_CHANNEL_BLEND_MODES: Readonly<
	Record<DecalChannel, DecalBlendMode>
> = {
	baseColor: "lerp",
	normal: "normal",
	roughness: "lerp",
	metalness: "lerp",
	emissive: "lerp",
	occlusion: "lerp",
	specular: "lerp",
	specularColor: "lerp",
	clearcoat: "lerp",
	clearcoatRoughness: "lerp",
	clearcoatNormal: "normal",
	sheenColor: "lerp",
	sheenRoughness: "lerp",
	transmission: "lerp",
	thickness: "lerp",
	iridescence: "lerp",
	iridescenceThickness: "lerp",
	anisotropy: "lerp",
};

export interface DecalParams extends NodeParams {
	material?: Material | null;
	receiverLayerMask?: number;
	priority?: number;
	opacity?: number;
	edgeFade?: number;
	channelBlendModes?: DecalChannelBlendModes;
}

/**
 * A box projector that modifies opaque receiver material channels before
 * lighting.
 */
export class Decal extends Node {
	/** Existing engine material reused as the source of decal textures/factors. */
	public material: Material | null;
	/** Bitmask of receiver render layers affected by this decal. */
	public receiverLayerMask: number;
	/** Stable ordering key; lower priority decals are applied first. */
	public priority: number;
	/** Global opacity multiplier used by all enabled decal channels. */
	public opacity: number;
	/** Local box edge fade distance in normalized projector coordinates. */
	public edgeFade: number;
	/** Per-channel blend modes overriding `DEFAULT_DECAL_CHANNEL_BLEND_MODES`. */
	public channelBlendModes: DecalChannelBlendModes;

	public constructor(params: DecalParams = {}) {
		super({ ...params, idPrefix: params.idPrefix ?? "decal" });
		this.material = params.material ?? null;
		this.receiverLayerMask = normalizeRenderLayerMask(
			params.receiverLayerMask ?? 1
		);
		this.priority = normalizeFiniteNumber(params.priority, 0);
		this.opacity = clamp01(params.opacity ?? 1);
		this.edgeFade = clamp01(params.edgeFade ?? 0);
		this.channelBlendModes = normalizeChannelBlendModes(
			params.channelBlendModes
		);
	}

	protected override _createCloneInstance(): this {
		return new Decal({
			material: this.material,
		}) as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.material = this.material;
		target.receiverLayerMask = this.receiverLayerMask;
		target.priority = this.priority;
		target.opacity = this.opacity;
		target.edgeFade = this.edgeFade;
		target.channelBlendModes = { ...this.channelBlendModes };
	}
}

/**
 * Resolves a decal channel mode from explicit overrides or defaults.
 *
 * @param modes - Optional channel override map.
 * @param channel - Channel to resolve.
 * @returns A supported blend mode for the requested channel.
 * @sideEffects None.
 */
export function resolveDecalChannelBlendMode(
	modes: DecalChannelBlendModes | undefined,
	channel: DecalChannel
): DecalBlendMode {
	const mode = modes?.[channel];
	return isDecalBlendMode(mode) ? mode : DEFAULT_DECAL_CHANNEL_BLEND_MODES[channel];
}

function normalizeChannelBlendModes(
	modes: DecalChannelBlendModes | undefined
): DecalChannelBlendModes {
	if (!modes) {
		return {};
	}
	const result: DecalChannelBlendModes = {};
	for (const channel of DECAL_CHANNELS) {
		const mode = modes[channel];
		if (isDecalBlendMode(mode)) {
			result[channel] = mode;
		}
	}
	return result;
}

function isDecalBlendMode(value: unknown): value is DecalBlendMode {
	return typeof value === "string" &&
		(DECAL_BLEND_MODES as readonly string[]).includes(value);
}

function normalizeFiniteNumber(value: number | undefined, fallback: number): number {
	return Number.isFinite(value) ? (value as number) : fallback;
}

function clamp01(value: number): number {
	if (!Number.isFinite(value)) {
		return 1;
	}
	return Math.max(0, Math.min(1, value));
}
