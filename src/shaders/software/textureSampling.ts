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
	if (!map || !map.data) return null;
	if (map.width <= 0 || map.height <= 0) return null;

	const uv = resolveSoftwareTextureUV(map, u, v);

	let tx = Math.floor(uv.u * map.width);
	let ty = Math.floor(uv.v * map.height);

	tx = Math.max(0, Math.min(map.width - 1, tx));
	ty = Math.max(0, Math.min(map.height - 1, ty));

	const idx = (ty * map.width + tx) << 2;
	if (map.colorSpace === "HDR" || map.colorSpace === "Linear") {
		const isFloat = map.data instanceof Float32Array;
		const colorScale = isFloat ? 255 : 1;
		const alphaRaw = map.data[idx + 3];
		return {
			r: Math.max(0, Math.min(255, (map.data[idx] ?? 0) * colorScale)),
			g: Math.max(0, Math.min(255, (map.data[idx + 1] ?? 0) * colorScale)),
			b: Math.max(0, Math.min(255, (map.data[idx + 2] ?? 0) * colorScale)),
			a:
				alphaRaw === undefined ? 1
				: isFloat ? clamp(alphaRaw)
				: clamp(alphaRaw / 255),
		};
	}

	const alpha = map.data[idx + 3] ?? 255;
	return {
		r: map.data[idx],
		g: map.data[idx + 1],
		b: map.data[idx + 2],
		a: alpha / 255,
	};
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

function resolveSoftwareTextureUV(
	map: NonNullable<TextureLike>,
	u: number,
	v: number
): { u: number; v: number } {
	let uu = u * map.repeat.x;
	let vv = v * map.repeat.y;

	if (map.rotation !== 0) {
		const c = Math.cos(map.rotation);
		const s = Math.sin(map.rotation);
		const ru = uu * c - vv * s;
		const rv = uu * s + vv * c;
		uu = ru;
		vv = rv;
	}

	uu += map.offset.x;
	vv += map.offset.y;

	if (map.wrapS === "Repeat") uu = uu - Math.floor(uu);
	else if (map.wrapS === "MirroredRepeat") {
		const iter = Math.floor(uu);
		uu = uu - iter;
		if (Math.abs(iter) % 2 === 1) uu = 1.0 - uu;
	} else uu = clamp(uu);

	if (map.wrapT === "Repeat") vv = vv - Math.floor(vv);
	else if (map.wrapT === "MirroredRepeat") {
		const iter = Math.floor(vv);
		vv = vv - iter;
		if (Math.abs(iter) % 2 === 1) vv = 1.0 - vv;
	} else vv = clamp(vv);

	return { u: uu, v: vv };
}
