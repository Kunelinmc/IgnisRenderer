/**
 * Deterministic procedural noise helpers for CPU-side generation.
 */

import { clamp, lerp } from "./Common";

const HASH_TO_UNIT = 1 / 0x100000000;
const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/**
 * Function signature for deterministic 2D noise samplers.
 */
export type Noise2DFunction = (x: number, y: number, seed?: number) => number;

/**
 * Options shared by octave-based fractal noise functions.
 */
export interface FractalNoiseOptions {
	/**
	 * Integer-compatible seed used to decorrelate generated noise.
	 */
	seed?: number;
	/**
	 * Number of detail layers to accumulate. Values below 1 are clamped to 1.
	 */
	octaves?: number;
	/**
	 * Initial coordinate multiplier applied before sampling.
	 */
	frequency?: number;
	/**
	 * Output amplitude applied after normalized octave accumulation.
	 */
	amplitude?: number;
	/**
	 * Frequency multiplier between octaves.
	 */
	lacunarity?: number;
	/**
	 * Amplitude multiplier between octaves.
	 */
	persistence?: number;
}

/**
 * Distance metrics supported by cellular noise.
 */
export type CellularDistanceMetric = "euclidean" | "manhattan" | "chebyshev";

/**
 * Options for 2D cellular noise sampling.
 */
export interface CellularNoiseOptions {
	/**
	 * Integer-compatible seed used to decorrelate generated feature points.
	 */
	seed?: number;
	/**
	 * Feature-point jitter inside each lattice cell, in the range [0, 1].
	 */
	jitter?: number;
	/**
	 * Distance metric used to select the nearest feature point.
	 */
	distanceMetric?: CellularDistanceMetric;
}

/**
 * Result returned by 2D cellular noise sampling.
 */
export interface CellularNoiseResult {
	/**
	 * Normalized inverse distance to the nearest feature point, in [0, 1].
	 */
	value: number;
	/**
	 * Raw distance to the nearest feature point under the selected metric.
	 */
	distance: number;
	/**
	 * X coordinate of the lattice cell that owns the nearest feature point.
	 */
	cellX: number;
	/**
	 * Y coordinate of the lattice cell that owns the nearest feature point.
	 */
	cellY: number;
}

/**
 * Options for generating a dense 2D noise map.
 */
export interface NoiseMap2DOptions {
	/**
	 * Integer-compatible seed passed to the noise sampler.
	 */
	seed?: number;
	/**
	 * X coordinate of the first sampled map cell.
	 */
	offsetX?: number;
	/**
	 * Y coordinate of the first sampled map cell.
	 */
	offsetY?: number;
	/**
	 * Coordinate step between adjacent generated samples.
	 */
	frequency?: number;
	/**
	 * Output multiplier applied after sampling and optional normalization.
	 */
	amplitude?: number;
	/**
	 * Noise sampler used for each map cell. It should return values in [-1, 1].
	 */
	sampler?: Noise2DFunction;
	/**
	 * Maps signed sampler output from [-1, 1] to [0, 1] before amplitude.
	 */
	normalize?: boolean;
}

/**
 * Samples deterministic lattice white noise in [0, 1) for a 1D coordinate.
 *
 * @param x Lattice coordinate. Fractional values are floored.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Deterministic pseudo-random value in [0, 1).
 */
export function hashNoise1D(x: number, seed: number = 0): number {
	return hashToUnit(seed, Math.floor(x));
}

/**
 * Samples deterministic lattice white noise in [0, 1) for a 2D coordinate.
 *
 * @param x Lattice X coordinate. Fractional values are floored.
 * @param y Lattice Y coordinate. Fractional values are floored.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Deterministic pseudo-random value in [0, 1).
 */
export function hashNoise2D(x: number, y: number, seed: number = 0): number {
	return hashToUnit(seed, Math.floor(x), Math.floor(y));
}

/**
 * Samples deterministic lattice white noise in [0, 1) for a 3D coordinate.
 *
 * @param x Lattice X coordinate. Fractional values are floored.
 * @param y Lattice Y coordinate. Fractional values are floored.
 * @param z Lattice Z coordinate. Fractional values are floored.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Deterministic pseudo-random value in [0, 1).
 */
export function hashNoise3D(
	x: number,
	y: number,
	z: number,
	seed: number = 0
): number {
	return hashToUnit(seed, Math.floor(x), Math.floor(y), Math.floor(z));
}

/**
 * Samples smooth signed 1D value noise.
 *
 * @param x Continuous X coordinate.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function valueNoise1D(x: number, seed: number = 0): number {
	const x0 = Math.floor(x);
	const x1 = x0 + 1;
	const tx = fade(x - x0);
	return lerp(signedHash(seed, x0), signedHash(seed, x1), tx);
}

/**
 * Samples smooth signed 2D value noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function valueNoise2D(
	x: number,
	y: number,
	seed: number = 0
): number {
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const x1 = x0 + 1;
	const y1 = y0 + 1;
	const tx = fade(x - x0);
	const ty = fade(y - y0);
	const a = lerp(signedHash(seed, x0, y0), signedHash(seed, x1, y0), tx);
	const b = lerp(signedHash(seed, x0, y1), signedHash(seed, x1, y1), tx);
	return lerp(a, b, ty);
}

/**
 * Samples smooth signed 3D value noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param z Continuous Z coordinate.
 * @param seed Integer-compatible seed used to decorrelate output.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function valueNoise3D(
	x: number,
	y: number,
	z: number,
	seed: number = 0
): number {
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const z0 = Math.floor(z);
	const x1 = x0 + 1;
	const y1 = y0 + 1;
	const z1 = z0 + 1;
	const tx = fade(x - x0);
	const ty = fade(y - y0);
	const tz = fade(z - z0);
	const x00 = lerp(signedHash(seed, x0, y0, z0), signedHash(seed, x1, y0, z0), tx);
	const x10 = lerp(signedHash(seed, x0, y1, z0), signedHash(seed, x1, y1, z0), tx);
	const x01 = lerp(signedHash(seed, x0, y0, z1), signedHash(seed, x1, y0, z1), tx);
	const x11 = lerp(signedHash(seed, x0, y1, z1), signedHash(seed, x1, y1, z1), tx);
	const y0Value = lerp(x00, x10, ty);
	const y1Value = lerp(x01, x11, ty);
	return lerp(y0Value, y1Value, tz);
}

/**
 * Samples signed 1D Perlin gradient noise.
 *
 * @param x Continuous X coordinate.
 * @param seed Integer-compatible seed used to decorrelate gradients.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function perlinNoise1D(x: number, seed: number = 0): number {
	const x0 = Math.floor(x);
	const x1 = x0 + 1;
	const tx = x - x0;
	const g0 = hashToUnit(seed, x0) < 0.5 ? -1 : 1;
	const g1 = hashToUnit(seed, x1) < 0.5 ? -1 : 1;
	return lerp(g0 * tx, g1 * (tx - 1), fade(tx));
}

/**
 * Samples signed 2D Perlin gradient noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param seed Integer-compatible seed used to decorrelate gradients.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function perlinNoise2D(
	x: number,
	y: number,
	seed: number = 0
): number {
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const x1 = x0 + 1;
	const y1 = y0 + 1;
	const tx = x - x0;
	const ty = y - y0;
	const u = fade(tx);
	const v = fade(ty);
	const n00 = gradientDot2D(seed, x0, y0, tx, ty);
	const n10 = gradientDot2D(seed, x1, y0, tx - 1, ty);
	const n01 = gradientDot2D(seed, x0, y1, tx, ty - 1);
	const n11 = gradientDot2D(seed, x1, y1, tx - 1, ty - 1);
	const nx0 = lerp(n00, n10, u);
	const nx1 = lerp(n01, n11, u);
	return clamp(lerp(nx0, nx1, v) * Math.SQRT2, -1, 1);
}

/**
 * Samples signed 3D Perlin gradient noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param z Continuous Z coordinate.
 * @param seed Integer-compatible seed used to decorrelate gradients.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function perlinNoise3D(
	x: number,
	y: number,
	z: number,
	seed: number = 0
): number {
	const x0 = Math.floor(x);
	const y0 = Math.floor(y);
	const z0 = Math.floor(z);
	const x1 = x0 + 1;
	const y1 = y0 + 1;
	const z1 = z0 + 1;
	const tx = x - x0;
	const ty = y - y0;
	const tz = z - z0;
	const u = fade(tx);
	const v = fade(ty);
	const w = fade(tz);
	const n000 = gradientDot3D(seed, x0, y0, z0, tx, ty, tz);
	const n100 = gradientDot3D(seed, x1, y0, z0, tx - 1, ty, tz);
	const n010 = gradientDot3D(seed, x0, y1, z0, tx, ty - 1, tz);
	const n110 = gradientDot3D(seed, x1, y1, z0, tx - 1, ty - 1, tz);
	const n001 = gradientDot3D(seed, x0, y0, z1, tx, ty, tz - 1);
	const n101 = gradientDot3D(seed, x1, y0, z1, tx - 1, ty, tz - 1);
	const n011 = gradientDot3D(seed, x0, y1, z1, tx, ty - 1, tz - 1);
	const n111 = gradientDot3D(seed, x1, y1, z1, tx - 1, ty - 1, tz - 1);
	const x00 = lerp(n000, n100, u);
	const x10 = lerp(n010, n110, u);
	const x01 = lerp(n001, n101, u);
	const x11 = lerp(n011, n111, u);
	const y0Value = lerp(x00, x10, v);
	const y1Value = lerp(x01, x11, v);
	return clamp(lerp(y0Value, y1Value, w), -1, 1);
}

/**
 * Samples signed 2D simplex gradient noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param seed Integer-compatible seed used to decorrelate gradients.
 * @returns Smooth deterministic value in approximately [-1, 1].
 */
export function simplexNoise2D(
	x: number,
	y: number,
	seed: number = 0
): number {
	const skew = (x + y) * 0.3660254037844386;
	const i = Math.floor(x + skew);
	const j = Math.floor(y + skew);
	const unskew = (i + j) * 0.21132486540518713;
	const x0 = x - (i - unskew);
	const y0 = y - (j - unskew);
	const i1 = x0 > y0 ? 1 : 0;
	const j1 = x0 > y0 ? 0 : 1;
	const x1 = x0 - i1 + 0.21132486540518713;
	const y1 = y0 - j1 + 0.21132486540518713;
	const x2 = x0 - 1 + 0.42264973081037427;
	const y2 = y0 - 1 + 0.42264973081037427;
	const n0 = simplexCorner2D(seed, i, j, x0, y0);
	const n1 = simplexCorner2D(seed, i + i1, j + j1, x1, y1);
	const n2 = simplexCorner2D(seed, i + 1, j + 1, x2, y2);
	return clamp((n0 + n1 + n2) * 70, -1, 1);
}

/**
 * Samples normalized 2D fractal Brownian motion from a base noise sampler.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param options Octave accumulation options.
 * @param sampler Signed base sampler. Defaults to `perlinNoise2D`.
 * @returns Signed fractal value in approximately [-amplitude, amplitude].
 */
export function fractalBrownianMotion2D(
	x: number,
	y: number,
	options: FractalNoiseOptions = {},
	sampler: Noise2DFunction = perlinNoise2D
): number {
	const state = resolveFractalOptions(options);
	let sum = 0;
	let maxAmplitude = 0;
	let frequency = state.frequency;
	let amplitude = 1;
	for (let octave = 0; octave < state.octaves; octave++) {
		sum += sampler(x * frequency, y * frequency, state.seed + octave) * amplitude;
		maxAmplitude += amplitude;
		frequency *= state.lacunarity;
		amplitude *= state.persistence;
	}
	return (sum / Math.max(maxAmplitude, 1e-8)) * state.amplitude;
}

/**
 * Samples normalized 2D turbulence noise from a base noise sampler.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param options Octave accumulation options.
 * @param sampler Signed base sampler. Defaults to `perlinNoise2D`.
 * @returns Non-negative turbulent value in approximately [0, amplitude].
 */
export function turbulenceNoise2D(
	x: number,
	y: number,
	options: FractalNoiseOptions = {},
	sampler: Noise2DFunction = perlinNoise2D
): number {
	const state = resolveFractalOptions(options);
	let sum = 0;
	let maxAmplitude = 0;
	let frequency = state.frequency;
	let amplitude = 1;
	for (let octave = 0; octave < state.octaves; octave++) {
		sum += Math.abs(
			sampler(x * frequency, y * frequency, state.seed + octave)
		) * amplitude;
		maxAmplitude += amplitude;
		frequency *= state.lacunarity;
		amplitude *= state.persistence;
	}
	return (sum / Math.max(maxAmplitude, 1e-8)) * state.amplitude;
}

/**
 * Samples normalized 2D ridged multifractal noise.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param options Octave accumulation options.
 * @param sampler Signed base sampler. Defaults to `perlinNoise2D`.
 * @returns Non-negative ridge value in approximately [0, amplitude].
 */
export function ridgedNoise2D(
	x: number,
	y: number,
	options: FractalNoiseOptions = {},
	sampler: Noise2DFunction = perlinNoise2D
): number {
	const state = resolveFractalOptions(options);
	let sum = 0;
	let maxAmplitude = 0;
	let frequency = state.frequency;
	let amplitude = 1;
	for (let octave = 0; octave < state.octaves; octave++) {
		const ridge = 1 - Math.abs(
			sampler(x * frequency, y * frequency, state.seed + octave)
		);
		sum += ridge * ridge * amplitude;
		maxAmplitude += amplitude;
		frequency *= state.lacunarity;
		amplitude *= state.persistence;
	}
	return (sum / Math.max(maxAmplitude, 1e-8)) * state.amplitude;
}

/**
 * Samples 2D cellular (Worley-style) noise using one feature point per cell.
 *
 * @param x Continuous X coordinate.
 * @param y Continuous Y coordinate.
 * @param options Cellular sampling options.
 * @returns Nearest-cell information and normalized inverse-distance value.
 */
export function cellularNoise2D(
	x: number,
	y: number,
	options: CellularNoiseOptions = {}
): CellularNoiseResult {
	const seed = sanitizeSeed(options.seed);
	const jitter = clamp(finiteOr(options.jitter, 1), 0, 1);
	const distanceMetric = options.distanceMetric ?? "euclidean";
	const baseX = Math.floor(x);
	const baseY = Math.floor(y);
	let nearestDistance = Number.POSITIVE_INFINITY;
	let nearestCellX = baseX;
	let nearestCellY = baseY;
	for (let yOffset = -1; yOffset <= 1; yOffset++) {
		for (let xOffset = -1; xOffset <= 1; xOffset++) {
			const cellX = baseX + xOffset;
			const cellY = baseY + yOffset;
			const featureX = cellX + featureOffset(seed, cellX, cellY, 0, jitter);
			const featureY = cellY + featureOffset(seed, cellX, cellY, 1, jitter);
			const distance = cellularDistance(
				x - featureX,
				y - featureY,
				distanceMetric
			);
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearestCellX = cellX;
				nearestCellY = cellY;
			}
		}
	}
	return {
		value: 1 - clamp(nearestDistance / maxCellularDistance(distanceMetric), 0, 1),
		distance: nearestDistance,
		cellX: nearestCellX,
		cellY: nearestCellY,
	};
}

/**
 * Generates a row-major 2D Float32 noise map.
 *
 * @param width Number of samples per row. Values below 1 are clamped to 1.
 * @param height Number of rows. Values below 1 are clamped to 1.
 * @param options Sampling, normalization, and amplitude options.
 * @returns Row-major Float32Array with `width * height` entries.
 */
export function generateNoiseMap2D(
	width: number,
	height: number,
	options: NoiseMap2DOptions = {}
): Float32Array {
	const safeWidth = Math.max(1, Math.floor(width));
	const safeHeight = Math.max(1, Math.floor(height));
	const seed = sanitizeSeed(options.seed);
	const offsetX = finiteOr(options.offsetX, 0);
	const offsetY = finiteOr(options.offsetY, 0);
	const frequency = finiteOr(options.frequency, 1);
	const amplitude = finiteOr(options.amplitude, 1);
	const sampler = options.sampler ?? perlinNoise2D;
	const data = new Float32Array(safeWidth * safeHeight);
	for (let y = 0; y < safeHeight; y++) {
		for (let x = 0; x < safeWidth; x++) {
			let value = sampler(offsetX + x * frequency, offsetY + y * frequency, seed);
			if (options.normalize) {
				value = clamp(value * 0.5 + 0.5, 0, 1);
			}
			data[y * safeWidth + x] = value * amplitude;
		}
	}
	return data;
}

interface ResolvedFractalNoiseOptions {
	seed: number;
	octaves: number;
	frequency: number;
	amplitude: number;
	lacunarity: number;
	persistence: number;
}

function resolveFractalOptions(
	options: FractalNoiseOptions
): ResolvedFractalNoiseOptions {
	return {
		seed: sanitizeSeed(options.seed),
		octaves: Math.max(1, Math.floor(finiteOr(options.octaves, 4))),
		frequency: finiteOr(options.frequency, 1),
		amplitude: finiteOr(options.amplitude, 1),
		lacunarity: finiteOr(options.lacunarity, 2),
		persistence: finiteOr(options.persistence, 0.5),
	};
}

function gradientDot2D(
	seed: number,
	latticeX: number,
	latticeY: number,
	dx: number,
	dy: number
): number {
	const angle = hashToUnit(seed, latticeX, latticeY) * Math.PI * 2;
	return Math.cos(angle) * dx + Math.sin(angle) * dy;
}

function simplexCorner2D(
	seed: number,
	latticeX: number,
	latticeY: number,
	dx: number,
	dy: number
): number {
	let t = 0.5 - dx * dx - dy * dy;
	if (t <= 0) return 0;
	t *= t;
	return t * t * gradientDot2D(seed, latticeX, latticeY, dx, dy);
}

function gradientDot3D(
	seed: number,
	latticeX: number,
	latticeY: number,
	latticeZ: number,
	dx: number,
	dy: number,
	dz: number
): number {
	const hash = hashUint(seed, latticeX, latticeY, latticeZ) & 15;
	const u = hash < 8 ? dx : dy;
	const v =
		hash < 4 ? dy
		: hash === 12 || hash === 14 ? dx
		: dz;
	return ((hash & 1) === 0 ? u : -u) + ((hash & 2) === 0 ? v : -v);
}

function featureOffset(
	seed: number,
	cellX: number,
	cellY: number,
	axis: number,
	jitter: number
): number {
	return 0.5 + (hashToUnit(seed, cellX, cellY, axis) - 0.5) * jitter;
}

function cellularDistance(
	dx: number,
	dy: number,
	metric: CellularDistanceMetric
): number {
	if (metric === "manhattan") {
		return Math.abs(dx) + Math.abs(dy);
	}
	if (metric === "chebyshev") {
		return Math.max(Math.abs(dx), Math.abs(dy));
	}
	return Math.hypot(dx, dy);
}

function maxCellularDistance(metric: CellularDistanceMetric): number {
	if (metric === "manhattan") return 2;
	if (metric === "chebyshev") return 1;
	return Math.SQRT2;
}

function signedHash(seed: number, ...coords: number[]): number {
	return hashToUnit(seed, ...coords) * 2 - 1;
}

function hashToUnit(seed: number, ...coords: number[]): number {
	return hashUint(seed, ...coords) * HASH_TO_UNIT;
}

function hashUint(seed: number, ...coords: number[]): number {
	let hash = Math.imul(FNV_OFFSET ^ sanitizeSeed(seed), FNV_PRIME);
	for (const coord of coords) {
		hash ^= mixSignedInt(coord);
		hash = Math.imul(hash, FNV_PRIME);
	}
	hash ^= hash >>> 16;
	hash = Math.imul(hash, 0x7feb352d);
	hash ^= hash >>> 15;
	hash = Math.imul(hash, 0x846ca68b);
	hash ^= hash >>> 16;
	return hash >>> 0;
}

function mixSignedInt(value: number): number {
	let x = Math.floor(finiteOr(value, 0)) | 0;
	x ^= x >>> 16;
	x = Math.imul(x, 0x7feb352d);
	x ^= x >>> 15;
	x = Math.imul(x, 0x846ca68b);
	x ^= x >>> 16;
	return x >>> 0;
}

function fade(value: number): number {
	const t = clamp(value, 0, 1);
	return t * t * t * (t * (t * 6 - 15) + 10);
}

function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function sanitizeSeed(seed: number | undefined): number {
	return Math.floor(finiteOr(seed, 0)) | 0;
}
