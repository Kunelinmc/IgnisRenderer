import type { RGBA } from "../../foundation/Color";
import { clamp } from "../../maths/Common";
import type { TextureLike } from "../../materials";

const softwareTextureDataCache = new WeakMap<
	NonNullable<TextureLike>,
	{
		version: number;
		data: ReturnType<NonNullable<TextureLike>["readPixelData"]>;
	}
>();

/**
 * @internal
 * Samples a software-rendered material texture with engine UV transforms.
 *
 * @param map - Texture to sample. `null` and empty textures return `null`.
 * @param u - Source U coordinate before texture transform.
 * @param v - Source V coordinate before texture transform.
 * @returns RGBA sample in 0..255 material units, with Float32 HDR RGB allowed
 * above 255, and alpha in 0..1; otherwise `null`.
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
	if (!map) return false;
	if (map.width <= 0 || map.height <= 0) return false;
	const data = resolveSoftwareTextureData(map);
	if (!data) return false;

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
		const isFloat = data instanceof Float32Array;
		const colorScale = isFloat ? 255 : 1;
		const alphaRaw = data[idx + 3];
		const maximum = map.colorSpace === "HDR" ? Number.POSITIVE_INFINITY : 255;
		out.r = Math.max(0, Math.min(maximum, (data[idx] ?? 0) * colorScale));
		out.g = Math.max(0, Math.min(maximum, (data[idx + 1] ?? 0) * colorScale));
		out.b = Math.max(0, Math.min(maximum, (data[idx + 2] ?? 0) * colorScale));
		out.a =
			alphaRaw === undefined ? 1
			: isFloat ? clamp(alphaRaw)
			: clamp(alphaRaw / 255);
		return true;
	}

	const alpha = data[idx + 3] ?? 255;
	out.r = data[idx];
	out.g = data[idx + 1];
	out.b = data[idx + 2];
	out.a = alpha / 255;
	return true;
}

/**
 * Returns whether a software texture has readable pixel data.
 *
 * @internal Owned by software material and shadow evaluation.
 */
export function hasSoftwareTextureData(
	map: TextureLike | undefined
): boolean {
	return !!map && resolveSoftwareTextureData(map) !== null;
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

function resolveSoftwareTextureData(
	map: NonNullable<TextureLike>
): ReturnType<NonNullable<TextureLike>["readPixelData"]> {
	if (map.data) {
		return map.data;
	}
	const cached = softwareTextureDataCache.get(map);
	if (cached?.version === map.version) {
		return cached.data;
	}
	const data = map.readPixelData(0);
	softwareTextureDataCache.set(map, {
		version: map.version,
		data,
	});
	return data;
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
