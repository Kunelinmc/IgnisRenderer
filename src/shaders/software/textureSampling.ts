import type { RGBA } from "../../foundation/Color";
import { clamp } from "../../maths/Common";
import type { TextureLike } from "../../materials";

/**
 * @internal
 * Samples a software-rendered material texture with engine UV transforms.
 *
 * @param map - Texture to sample. `null` and empty textures return `null`.
 * @param u - Source U coordinate before texture transform.
 * @param v - Source V coordinate before texture transform.
 * @returns RGBA sample in 0..255 RGB units and 0..1 alpha, or `null`.
 * @sideEffects None.
 */
export function sampleSoftwareTextureMap(
	map: TextureLike | undefined,
	u: number,
	v: number
): RGBA | null {
	const sample: RGBA = { r: 0, g: 0, b: 0, a: 1 };
	return sampleSoftwareTextureMapInto(map, u, v, sample) ? sample : null;
}

/**
 * Samples into caller-owned storage so fragment hot paths can avoid allocations.
 *
 * @internal Owned by software material and decal evaluation.
 */
export function sampleSoftwareTextureMapInto(
	map: TextureLike | undefined,
	u: number,
	v: number,
	out: RGBA
): boolean {
	if (!map || !map.data) return false;
	if (map.width <= 0 || map.height <= 0) return false;

	let uu = u * map.repeat.x;
	let vv = v * map.repeat.y;

	if (map.rotation !== 0) {
		const c = Math.cos(map.rotation);
		const s = Math.sin(map.rotation);
		const ru = uu * c - vv * s;
		vv = uu * s + vv * c;
		uu = ru;
	}
	uu += map.offset.x;
	vv += map.offset.y;
	uu = wrapSoftwareTextureCoordinate(uu, map.wrapS);
	vv = wrapSoftwareTextureCoordinate(vv, map.wrapT);

	let tx = Math.floor(uu * map.width);
	let ty = Math.floor(vv * map.height);

	tx = Math.max(0, Math.min(map.width - 1, tx));
	ty = Math.max(0, Math.min(map.height - 1, ty));

	const idx = (ty * map.width + tx) << 2;
	if (map.colorSpace === "HDR" || map.colorSpace === "Linear") {
		const isFloat = map.data instanceof Float32Array;
		const colorScale = isFloat ? 255 : 1;
		const alphaRaw = map.data[idx + 3];
		out.r = Math.max(0, Math.min(255, (map.data[idx] ?? 0) * colorScale));
		out.g = Math.max(0, Math.min(255, (map.data[idx + 1] ?? 0) * colorScale));
		out.b = Math.max(0, Math.min(255, (map.data[idx + 2] ?? 0) * colorScale));
		out.a =
			alphaRaw === undefined ? 1
			: isFloat ? clamp(alphaRaw)
			: clamp(alphaRaw / 255);
		return true;
	}

	const alpha = map.data[idx + 3] ?? 255;
	out.r = map.data[idx];
	out.g = map.data[idx + 1];
	out.b = map.data[idx + 2];
	out.a = alpha / 255;
	return true;
}

/**
 * @internal
 * Samples only alpha from a software-rendered material texture.
 *
 * @param map - Texture to sample.
 * @param u - Source U coordinate before texture transform.
 * @param v - Source V coordinate before texture transform.
 * @returns Alpha in 0..1, defaulting to 1 for missing alpha channels.
 * @sideEffects None.
 */
export function sampleSoftwareTextureAlpha(
	map: TextureLike | undefined,
	u: number,
	v: number
): number {
	const sample = sampleSoftwareTextureMap(map, u, v);
	return sample ? sample.a : 1;
}

function wrapSoftwareTextureCoordinate(
	value: number,
	mode: NonNullable<TextureLike>["wrapS"]
): number {
	if (mode === "Repeat") return value - Math.floor(value);
	if (mode === "MirroredRepeat") {
		const iteration = Math.floor(value);
		const fraction = value - iteration;
		return Math.abs(iteration) % 2 === 1 ? 1 - fraction : fraction;
	}
	return clamp(value);
}
