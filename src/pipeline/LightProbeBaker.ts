import { SH } from "../maths/SH";
import { Vector3 } from "../maths/Vector3";
import { hammersley, importanceSampleGGX_VNDF } from "../maths/Sampling";
import { lerp, sRGBToLinear } from "../maths/Common";
import type { IVector3 } from "../maths/types";
import { Texture } from "../core/Texture";
import { LightProbe } from "../lights/LightProbe";

export function bakeLightProbeFromEnvironmentMap(envMap: Texture): LightProbe {
	const probe = projectEquirectToSH(envMap);
	probe.prefilteredMap = prefilterEnvMap(envMap);
	return probe;
}

function projectEquirectToSH(envMap: Texture): LightProbe {
	if (!envMap || !envMap.data) {
		return new LightProbe();
	}

	const { width, height, data } = envMap;
	const sh = SH.empty();

	const MAX_WIDTH = 128;
	const MAX_HEIGHT = 64;

	const sampleWidth = Math.min(width, MAX_WIDTH);
	const sampleHeight = Math.min(height, MAX_HEIGHT);

	const stepX = width / sampleWidth;
	const stepY = height / sampleHeight;

	const dTheta = Math.PI / sampleHeight;
	const dPhi = (2 * Math.PI) / sampleWidth;

	let totalWeight = 0;

	for (let sj = 0; sj < sampleHeight; sj++) {
		const theta = (sj + 0.5) * dTheta;
		const sinTheta = Math.sin(theta);
		const cosTheta = Math.cos(theta);
		const weight = sinTheta * dTheta * dPhi;

		const j = Math.floor((sj + 0.5) * stepY);

		for (let si = 0; si < sampleWidth; si++) {
			const phi = (si + 0.5) * dPhi;

			const x = sinTheta * Math.sin(phi);
			const y = cosTheta;
			const z = sinTheta * Math.cos(phi);

			const basis = SH.evalBasis({ x, y, z });

			const i = Math.floor((si + 0.5) * stepX);
			const idx = (j * width + i) * 4;

			const isLinear =
				envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";
			const r = isLinear
				? data[idx] * 255
				: sRGBToLinear(data[idx] / 255) * 255;
			const g = isLinear
				? data[idx + 1] * 255
				: sRGBToLinear(data[idx + 1] / 255) * 255;
			const b = isLinear
				? data[idx + 2] * 255
				: sRGBToLinear(data[idx + 2] / 255) * 255;

			for (let k = 0; k < sh.length; k++) {
				const bK = basis[k] * weight;
				sh[k].r += r * bK;
				sh[k].g += g * bK;
				sh[k].b += b * bK;
			}

			totalWeight += weight;
		}
	}

	const normFactor = (4 * Math.PI) / totalWeight;

	for (let k = 0; k < sh.length; k++) {
		sh[k].r *= normFactor;
		sh[k].g *= normFactor;
		sh[k].b *= normFactor;
	}

	return new LightProbe(sh);
}

function prefilterEnvMap(envMap: Texture): Texture {
	const baseWidth = Math.min(envMap.width, 128);
	const baseHeight = Math.min(envMap.height, 64);
	const maxMipLevels = 5;

	const prefiltered = new Texture(null, baseWidth, baseHeight, "HDR");
	prefiltered.mipmaps = [];

	for (let level = 0; level < maxMipLevels; level++) {
		const roughness = level / (maxMipLevels - 1);
		const sampleCount = Math.floor(lerp(1024, 64, roughness));
		const w = Math.max(1, baseWidth >> level);
		const h = Math.max(1, baseHeight >> level);
		const data = new Float32Array(w * h * 4);

		for (let j = 0; j < h; j++) {
			const theta = ((j + 0.5) / h) * Math.PI;
			for (let i = 0; i < w; i++) {
				const phi = ((i + 0.5) / w) * 2 * Math.PI;

				const normal = {
					x: Math.sin(theta) * Math.sin(phi),
					y: Math.cos(theta),
					z: Math.sin(theta) * Math.cos(phi),
				};

				const radiance = prefilterSpecular(
					envMap,
					normal,
					roughness,
					sampleCount
				);
				const idx = (j * w + i) * 4;
				data[idx] = radiance.r;
				data[idx + 1] = radiance.g;
				data[idx + 2] = radiance.b;
				data[idx + 3] = 1.0;
			}
		}

		prefiltered.mipmaps.push(data);
	}

	prefiltered.data = prefiltered.mipmaps[0];
	return prefiltered;
}

function prefilterSpecular(
	envMap: Texture,
	normal: IVector3,
	roughness: number,
	sampleCount: number
): { r: number; g: number; b: number } {
	let totalWeight = 0;
	let prefilteredColor = { r: 0, g: 0, b: 0 };

	for (let i = 0; i < sampleCount; i++) {
		const xi = hammersley(i, sampleCount);
		const view = normal;
		const half = importanceSampleGGX_VNDF(xi, view, normal, roughness);
		const nDotH = Math.max(Vector3.dot(normal, half), 0);
		const lightDir = Vector3.normalize({
			x: 2.0 * nDotH * half.x - view.x,
			y: 2.0 * nDotH * half.y - view.y,
			z: 2.0 * nDotH * half.z - view.z,
		});

		const nDotL = Math.max(Vector3.dot(normal, lightDir), 0);
		if (nDotL <= 0) continue;

		const phi = Math.atan2(lightDir.x, lightDir.z);
		const theta = Math.acos(Math.max(-1, Math.min(1, lightDir.y)));
		const u = (phi + Math.PI) / (2 * Math.PI);
		const v = theta / Math.PI;

		const sample = envMap.sample(u, v);
		const isLinear =
			envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";

		const r = isLinear ? sample.r / 255 : sRGBToLinear(sample.r / 255);
		const g = isLinear ? sample.g / 255 : sRGBToLinear(sample.g / 255);
		const b = isLinear ? sample.b / 255 : sRGBToLinear(sample.b / 255);

		prefilteredColor.r += r * nDotL;
		prefilteredColor.g += g * nDotL;
		prefilteredColor.b += b * nDotL;
		totalWeight += nDotL;
	}

	if (totalWeight > 0) {
		prefilteredColor.r /= totalWeight;
		prefilteredColor.g /= totalWeight;
		prefilteredColor.b /= totalWeight;
	}

	return prefilteredColor;
}
