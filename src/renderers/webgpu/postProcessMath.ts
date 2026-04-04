export const TAA_HALTON_SAMPLE_COUNT = 16;

export function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function ceilDiv(value: number, divisor: number): number {
	return Math.max(1, Math.ceil(value / Math.max(divisor, 1)));
}

export function halton(index: number, base: number): number {
	if (base <= 1) {
		throw new Error(`Halton base must be > 1, received ${base}`);
	}
	let result = 0;
	let fraction = 1 / base;
	let i = Math.max(0, Math.floor(index));
	while (i > 0) {
		result += fraction * (i % base);
		i = Math.floor(i / base);
		fraction /= base;
	}
	return result;
}

export function computeHaltonJitterNDC(
	frameIndex: number,
	width: number,
	height: number,
	jitterScale: number = 1,
	sampleCount: number = TAA_HALTON_SAMPLE_COUNT
): [number, number] {
	const safeWidth = Math.max(1, Math.floor(width));
	const safeHeight = Math.max(1, Math.floor(height));
	const safeSampleCount = Math.max(1, Math.floor(sampleCount));
	const clampedScale =
		Number.isFinite(jitterScale) && jitterScale >= 0 ? jitterScale : 1;
	const safeFrameIndex = Math.max(0, Math.floor(frameIndex));
	const sampleIndex = (safeFrameIndex % safeSampleCount) + 1;
	const cycleIndex = Math.floor(safeFrameIndex / safeSampleCount);
	const rotationX = fract(cycleIndex * 0.7548776662466927);
	const rotationY = fract(cycleIndex * 0.5698402909980532);
	const jitterX = wrapUnit(halton(sampleIndex, 2) + rotationX) - 0.5;
	const jitterY = wrapUnit(halton(sampleIndex, 3) + rotationY) - 0.5;
	return [
		(jitterX * 2 * clampedScale) / safeWidth,
		(jitterY * 2 * clampedScale) / safeHeight,
	];
}

export function reprojectHistoryUv(
	uv: [number, number],
	motion: [number, number]
): [number, number] {
	return [uv[0] - motion[0] * 0.5, uv[1] + motion[1] * 0.5];
}

export function isDepthHistoryCompatible(
	currentDepth: number,
	previousDepth: number,
	threshold: number
): boolean {
	const safeThreshold =
		Number.isFinite(threshold) && threshold >= 0 ? threshold : 0;
	if (!(currentDepth > 0) || !(previousDepth > 0)) return false;
	const relativeDiff =
		Math.abs(currentDepth - previousDepth) /
		Math.max(Math.max(currentDepth, previousDepth), 1e-4);
	return relativeDiff <= safeThreshold;
}

export function rgbToYCoCg(
	rgb: [number, number, number]
): [number, number, number] {
	const co = rgb[0] - rgb[2];
	const t = rgb[2] + co * 0.5;
	const cg = rgb[1] - t;
	const y = t + cg * 0.5;
	return [y, co, cg];
}

export function yCoCgToRgb(
	yCoCg: [number, number, number]
): [number, number, number] {
	const t = yCoCg[0] - yCoCg[2] * 0.5;
	const g = yCoCg[2] + t;
	const b = t - yCoCg[1] * 0.5;
	const r = b + yCoCg[1];
	return [r, g, b];
}

export function clampHistoryToNeighborhoodYCoCg(
	historyRgb: [number, number, number],
	neighborhood: [number, number, number][],
	varianceClampGamma: number
): [number, number, number] {
	if (neighborhood.length === 0) {
		return historyRgb;
	}
	const gamma =
		Number.isFinite(varianceClampGamma) && varianceClampGamma >= 0 ?
			varianceClampGamma
		:	0;
	let minY = Number.POSITIVE_INFINITY;
	let minCo = Number.POSITIVE_INFINITY;
	let minCg = Number.POSITIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;
	let maxCo = Number.NEGATIVE_INFINITY;
	let maxCg = Number.NEGATIVE_INFINITY;
	let sumY = 0;
	let sumCo = 0;
	let sumCg = 0;
	let sumSqY = 0;
	let sumSqCo = 0;
	let sumSqCg = 0;
	for (const rgb of neighborhood) {
		const ycocg = rgbToYCoCg(rgb);
		minY = Math.min(minY, ycocg[0]);
		minCo = Math.min(minCo, ycocg[1]);
		minCg = Math.min(minCg, ycocg[2]);
		maxY = Math.max(maxY, ycocg[0]);
		maxCo = Math.max(maxCo, ycocg[1]);
		maxCg = Math.max(maxCg, ycocg[2]);
		sumY += ycocg[0];
		sumCo += ycocg[1];
		sumCg += ycocg[2];
		sumSqY += ycocg[0] * ycocg[0];
		sumSqCo += ycocg[1] * ycocg[1];
		sumSqCg += ycocg[2] * ycocg[2];
	}
	const invCount = 1 / neighborhood.length;
	const meanY = sumY * invCount;
	const meanCo = sumCo * invCount;
	const meanCg = sumCg * invCount;
	const sigmaY = Math.sqrt(Math.max(sumSqY * invCount - meanY * meanY, 0));
	const sigmaCo = Math.sqrt(Math.max(sumSqCo * invCount - meanCo * meanCo, 0));
	const sigmaCg = Math.sqrt(Math.max(sumSqCg * invCount - meanCg * meanCg, 0));
	const varMinY = meanY - sigmaY * gamma;
	const varMinCo = meanCo - sigmaCo * gamma;
	const varMinCg = meanCg - sigmaCg * gamma;
	const varMaxY = meanY + sigmaY * gamma;
	const varMaxCo = meanCo + sigmaCo * gamma;
	const varMaxCg = meanCg + sigmaCg * gamma;
	const clipMinY = Math.max(minY, varMinY);
	const clipMinCo = Math.max(minCo, varMinCo);
	const clipMinCg = Math.max(minCg, varMinCg);
	const clipMaxY = Math.min(maxY, varMaxY);
	const clipMaxCo = Math.min(maxCo, varMaxCo);
	const clipMaxCg = Math.min(maxCg, varMaxCg);
	const historyYCoCg = rgbToYCoCg(historyRgb);
	const clamped: [number, number, number] = [
		clamp(
			historyYCoCg[0],
			Math.min(clipMinY, clipMaxY),
			Math.max(clipMinY, clipMaxY)
		),
		clamp(
			historyYCoCg[1],
			Math.min(clipMinCo, clipMaxCo),
			Math.max(clipMinCo, clipMaxCo)
		),
		clamp(
			historyYCoCg[2],
			Math.min(clipMinCg, clipMaxCg),
			Math.max(clipMinCg, clipMaxCg)
		),
	];
	const rgb = yCoCgToRgb(clamped);
	return [Math.max(0, rgb[0]), Math.max(0, rgb[1]), Math.max(0, rgb[2])];
}

export interface HiZMipLevel {
	width: number;
	height: number;
	data: Float32Array;
}

export function buildHiZMipMinLevel(level: HiZMipLevel): HiZMipLevel {
	const outWidth = Math.max(1, level.width >> 1);
	const outHeight = Math.max(1, level.height >> 1);
	const out = new Float32Array(outWidth * outHeight);
	for (let y = 0; y < outHeight; y++) {
		for (let x = 0; x < outWidth; x++) {
			const x0 = clampInt(x * 2, 0, level.width - 1);
			const x1 = clampInt(x * 2 + 1, 0, level.width - 1);
			const y0 = clampInt(y * 2, 0, level.height - 1);
			const y1 = clampInt(y * 2 + 1, 0, level.height - 1);
			const d00 = level.data[y0 * level.width + x0];
			const d10 = level.data[y0 * level.width + x1];
			const d01 = level.data[y1 * level.width + x0];
			const d11 = level.data[y1 * level.width + x1];
			out[y * outWidth + x] = minPositive(
				minPositive(d00, d10),
				minPositive(d01, d11)
			);
		}
	}
	return {
		width: outWidth,
		height: outHeight,
		data: out,
	};
}

export function buildHiZMinChain(
	depth: Float32Array,
	width: number,
	height: number
): HiZMipLevel[] {
	const levels: HiZMipLevel[] = [
		{
			width: Math.max(1, Math.floor(width)),
			height: Math.max(1, Math.floor(height)),
			data: depth.slice(),
		},
	];
	while (
		levels[levels.length - 1].width > 1 ||
		levels[levels.length - 1].height > 1
	) {
		levels.push(buildHiZMipMinLevel(levels[levels.length - 1]));
	}
	return levels;
}

export function traceSSRDepthHit(
	depth: HiZMipLevel,
	originUv: [number, number],
	directionUv: [number, number],
	originDepth: number,
	stepSize: number,
	maxSteps: number,
	thickness: number
): { hit: boolean; hitUv: [number, number] | null } {
	const safeStep = Math.max(stepSize, 1e-4);
	const safeThickness = Math.max(0, thickness);
	let uvX = originUv[0];
	let uvY = originUv[1];
	let rayDepth = originDepth;
	for (let i = 0; i < Math.max(1, Math.floor(maxSteps)); i++) {
		uvX += directionUv[0] * safeStep;
		uvY += directionUv[1] * safeStep;
		rayDepth += safeStep;
		if (uvX < 0 || uvX > 1 || uvY < 0 || uvY > 1) {
			return { hit: false, hitUv: null };
		}
		const sceneDepth = sampleDepthNearest(depth, uvX, uvY);
		if (sceneDepth > 0 && rayDepth >= sceneDepth - safeThickness) {
			return { hit: true, hitUv: [uvX, uvY] };
		}
	}
	return { hit: false, hitUv: null };
}

function sampleDepthNearest(level: HiZMipLevel, u: number, v: number): number {
	const x = clampInt(Math.round(u * (level.width - 1)), 0, level.width - 1);
	const y = clampInt(Math.round(v * (level.height - 1)), 0, level.height - 1);
	return level.data[y * level.width + x];
}

function minPositive(a: number, b: number): number {
	if (a <= 0) return Math.max(0, b);
	if (b <= 0) return Math.max(0, a);
	return Math.min(a, b);
}

function clamp(value: number, minValue: number, maxValue: number): number {
	return Math.min(Math.max(value, minValue), maxValue);
}

function clampInt(value: number, minValue: number, maxValue: number): number {
	return Math.min(Math.max(value | 0, minValue), maxValue);
}

function fract(value: number): number {
	return value - Math.floor(value);
}

function wrapUnit(value: number): number {
	const f = fract(value);
	return f < 0 ? f + 1 : f;
}
