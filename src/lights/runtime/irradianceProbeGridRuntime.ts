import { Matrix4 } from "../../maths/Matrix4";
import { SH } from "../../maths/SH";
import type { IVector3, SHCoefficients } from "../../maths/types";
import {
	IrradianceProbeGrid,
	LightType,
	type SceneLight,
} from "../../lights";

export const MAX_ACTIVE_IRRADIANCE_PROBE_GRIDS = 1;
export const IRRADIANCE_PROBE_GRID_WEIGHT_EPSILON = 1e-6;
const IRRADIANCE_PROBE_GRID_BLEND_EPSILON = 1e-5;

export interface IrradianceProbeGridSample {
	grid: IrradianceProbeGrid | null;
	coverage: number;
	sh: SHCoefficients;
}

interface RankedIrradianceProbeGrid {
	grid: IrradianceProbeGrid;
	distanceSq: number;
	priority: number;
}

export function collectIrradianceProbeGrids(
	lights: SceneLight[]
): IrradianceProbeGrid[] {
	const grids: IrradianceProbeGrid[] = [];
	for (const light of lights) {
		if (light.type !== LightType.IrradianceProbeGrid) continue;
		const grid = light as IrradianceProbeGrid;
		grid.getRuntimeCache();
		grids.push(grid);
	}
	grids.sort(compareGridId);
	return grids;
}

export function refreshIrradianceProbeGridCaches(lights: SceneLight[]): void {
	for (const light of lights) {
		if (light.type !== LightType.IrradianceProbeGrid) continue;
		(light as IrradianceProbeGrid).refreshRuntimeCache();
	}
}

export function selectActiveIrradianceProbeGrid(
	lights: SceneLight[],
	cameraWorldPosition: IVector3 | null = null
): IrradianceProbeGrid | null {
	const grids = collectIrradianceProbeGrids(lights);
	if (grids.length <= 0) {
		return null;
	}
	if (!cameraWorldPosition) {
		grids.sort(compareActiveGridWithoutCamera);
		return grids[0] ?? null;
	}
	const ranked = grids.map((grid) => {
		const cache = grid.getRuntimeCache();
		const center = Matrix4.transformPoint(cache.gridToWorldMatrix, {
			x: 0,
			y: 0,
			z: 0,
		});
		const dx = center.x - cameraWorldPosition.x;
		const dy = center.y - cameraWorldPosition.y;
		const dz = center.z - cameraWorldPosition.z;
		return {
			grid,
			distanceSq: dx * dx + dy * dy + dz * dz,
			priority: cache.priority,
		};
	});
	ranked.sort(compareRankedGrid);
	return ranked[0]?.grid ?? null;
}

export function sampleActiveIrradianceProbeGrid(
	lights: SceneLight[],
	worldPosition: IVector3,
	cameraWorldPosition: IVector3 | null = null,
	out?: SHCoefficients
): IrradianceProbeGridSample {
	const grid = selectActiveIrradianceProbeGrid(lights, cameraWorldPosition);
	return sampleIrradianceProbeGrid(grid, worldPosition, out);
}

export function sampleIrradianceProbeGrid(
	grid: IrradianceProbeGrid | null,
	worldPosition: IVector3,
	out?: SHCoefficients
): IrradianceProbeGridSample {
	const target = clearSH(out ?? SH.empty());
	if (!grid) {
		return { grid: null, coverage: 0, sh: target };
	}

	const cache = grid.getRuntimeCache();
	if (cache.cellCount <= 0) {
		return { grid, coverage: 0, sh: target };
	}

	const local = Matrix4.transformPoint(cache.worldToGridMatrix, worldPosition);
	const metric = Math.max(
		Math.abs(local.x) * cache.invHalfExtents.x,
		Math.abs(local.y) * cache.invHalfExtents.y,
		Math.abs(local.z) * cache.invHalfExtents.z
	);
	const coverage = computeIrradianceProbeGridRawWeight(
		metric,
		cache.effectiveBlendDistance
	);
	if (coverage <= IRRADIANCE_PROBE_GRID_WEIGHT_EPSILON) {
		return { grid, coverage: 0, sh: target };
	}

	const dims = cache.dimensions;
	const gx = resolveGridSampleAxis(
		local.x,
		cache.halfExtents.x,
		dims.x
	);
	const gy = resolveGridSampleAxis(
		local.y,
		cache.halfExtents.y,
		dims.y
	);
	const gz = resolveGridSampleAxis(
		local.z,
		cache.halfExtents.z,
		dims.z
	);
	const x0 = Math.floor(gx);
	const y0 = Math.floor(gy);
	const z0 = Math.floor(gz);
	const x1 = Math.min(dims.x - 1, x0 + 1);
	const y1 = Math.min(dims.y - 1, y0 + 1);
	const z1 = Math.min(dims.z - 1, z0 + 1);
	const tx = gx - x0;
	const ty = gy - y0;
	const tz = gz - z0;
	let totalWeight = 0;

	totalWeight += accumulateCellSH(
		grid,
		target,
		x0,
		y0,
		z0,
		(1 - tx) * (1 - ty) * (1 - tz)
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x1,
		y0,
		z0,
		tx * (1 - ty) * (1 - tz)
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x0,
		y1,
		z0,
		(1 - tx) * ty * (1 - tz)
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x1,
		y1,
		z0,
		tx * ty * (1 - tz)
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x0,
		y0,
		z1,
		(1 - tx) * (1 - ty) * tz
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x1,
		y0,
		z1,
		tx * (1 - ty) * tz
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x0,
		y1,
		z1,
		(1 - tx) * ty * tz
	);
	totalWeight += accumulateCellSH(
		grid,
		target,
		x1,
		y1,
		z1,
		tx * ty * tz
	);

	if (totalWeight <= IRRADIANCE_PROBE_GRID_WEIGHT_EPSILON) {
		clearSH(target);
		return { grid, coverage: 0, sh: target };
	}

	const invWeight = 1 / totalWeight;
	for (let i = 0; i < target.length; i++) {
		target[i].r *= invWeight;
		target[i].g *= invWeight;
		target[i].b *= invWeight;
	}
	return {
		grid,
		coverage: clamp(coverage, 0, 1),
		sh: target,
	};
}

export function computeIrradianceProbeGridRawWeight(
	metric: number,
	effectiveBlendDistance: number
): number {
	if (!Number.isFinite(metric)) return 0;
	const safeBlendDistance = Math.max(
		effectiveBlendDistance,
		IRRADIANCE_PROBE_GRID_BLEND_EPSILON
	);
	const x = clamp((metric - 1) / safeBlendDistance, 0, 1);
	return 1 - smoothstep(0, 1, x);
}

function resolveGridSampleAxis(
	localValue: number,
	halfExtent: number,
	dimension: number
): number {
	if (dimension <= 1) {
		return 0;
	}
	const normalized = clamp(
		localValue / Math.max(halfExtent, 1e-6) * 0.5 + 0.5,
		0,
		1
	);
	return normalized * (dimension - 1);
}

function accumulateCellSH(
	grid: IrradianceProbeGrid,
	target: SHCoefficients,
	x: number,
	y: number,
	z: number,
	weight: number
): number {
	if (weight <= IRRADIANCE_PROBE_GRID_WEIGHT_EPSILON) {
		return 0;
	}
	const index = grid.getCellIndex(x, y, z);
	if (!grid.isCellValid(index)) {
		return 0;
	}
	const coeffs = grid.getCellSH(index);
	for (let i = 0; i < target.length; i++) {
		const coeff = coeffs[i];
		target[i].r += (coeff?.r ?? 0) * weight;
		target[i].g += (coeff?.g ?? 0) * weight;
		target[i].b += (coeff?.b ?? 0) * weight;
	}
	return weight;
}

function clearSH(target: SHCoefficients): SHCoefficients {
	for (const coeff of target) {
		coeff.r = 0;
		coeff.g = 0;
		coeff.b = 0;
	}
	return target;
}

function compareRankedGrid(
	left: RankedIrradianceProbeGrid,
	right: RankedIrradianceProbeGrid
): number {
	if (left.priority !== right.priority) {
		return right.priority - left.priority;
	}
	if (left.distanceSq !== right.distanceSq) {
		return left.distanceSq - right.distanceSq;
	}
	return compareGridId(left.grid, right.grid);
}

function compareActiveGridWithoutCamera(
	left: IrradianceProbeGrid,
	right: IrradianceProbeGrid
): number {
	const leftPriority = left.getRuntimeCache().priority;
	const rightPriority = right.getRuntimeCache().priority;
	if (leftPriority !== rightPriority) {
		return rightPriority - leftPriority;
	}
	return compareGridId(left, right);
}

function compareGridId(
	left: IrradianceProbeGrid,
	right: IrradianceProbeGrid
): number {
	return left.id.localeCompare(right.id);
}

function clamp(value: number, min: number, max: number): number {
	if (value <= min) return min;
	if (value >= max) return max;
	return value;
}

function smoothstep(edge0: number, edge1: number, x: number): number {
	const t = clamp((x - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
	return t * t * (3 - 2 * t);
}
