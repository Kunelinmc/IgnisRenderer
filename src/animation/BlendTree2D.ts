import type { BlendTreeChildWeight } from "./types";

const BLEND_EPSILON = 1e-6;
const TRIANGLE_EPSILON = 1e-8;

export type BlendTree2DBlendMode = "directional" | "triangulation";

export interface BlendTree2DChild {
	clipName: string;
	positionX: number;
	positionY: number;
}

export interface BlendTree2DOptions {
	name: string;
	parameterX: string;
	parameterY: string;
	children: BlendTree2DChild[];
	blendMode?: BlendTree2DBlendMode;
}

interface BlendTree2DTriangle {
	indices: [number, number, number];
	area2: number;
}

export class BlendTree2D {
	public readonly name: string;
	public readonly parameterX: string;
	public readonly parameterY: string;
	public readonly children: BlendTree2DChild[];
	public readonly blendMode: BlendTree2DBlendMode;

	private readonly _triangles: BlendTree2DTriangle[];

	constructor(options: BlendTree2DOptions) {
		this.name = options.name;
		this.parameterX = options.parameterX;
		this.parameterY = options.parameterY;
		this.children = [...options.children];
		this.blendMode = options.blendMode ?? "directional";
		this._triangles = this._buildTriangles();
	}

	public evaluate(
		parameterXValue: number,
		parameterYValue: number,
		blendMode: BlendTree2DBlendMode = this.blendMode
	): BlendTreeChildWeight[] {
		if (this.children.length === 0) return [];
		if (this.children.length === 1) {
			return [{ clipName: this.children[0].clipName, weight: 1 }];
		}

		const exactMatch = this._findExactMatch(parameterXValue, parameterYValue);
		if (exactMatch >= 0) {
			return [{ clipName: this.children[exactMatch].clipName, weight: 1 }];
		}

		if (blendMode === "triangulation") {
			return this._evaluateTriangulation(parameterXValue, parameterYValue);
		}
		return this._evaluateDirectional(parameterXValue, parameterYValue);
	}

	private _evaluateDirectional(
		parameterXValue: number,
		parameterYValue: number
	): BlendTreeChildWeight[] {
		const queryLength = Math.hypot(parameterXValue, parameterYValue);
		const safeQueryLength = Math.max(queryLength, BLEND_EPSILON);
		const maxChildRadius = this._maxChildRadius();

		const rawWeights = new Array<number>(this.children.length).fill(0);
		let totalWeight = 0;

		for (let i = 0; i < this.children.length; i++) {
			const child = this.children[i];
			const childLength = Math.hypot(child.positionX, child.positionY);
			let weight = 0;

			if (childLength <= BLEND_EPSILON) {
				if (queryLength <= BLEND_EPSILON) {
					weight = 1;
				} else {
					const centerRadius = Math.max(maxChildRadius, 1);
					weight = Math.max(0, 1 - queryLength / centerRadius);
				}
			} else if (queryLength <= BLEND_EPSILON) {
				weight = 1 / childLength;
			} else {
				const dot =
					(parameterXValue * child.positionX +
						parameterYValue * child.positionY) /
					(safeQueryLength * childLength);
				if (dot > 0) {
					const directionalWeight = dot * dot;
					const radialDistance = Math.abs(queryLength - childLength);
					const radialWeight =
						1 / (radialDistance + childLength * 0.25 + BLEND_EPSILON);
					weight = directionalWeight * radialWeight;
				}
			}

			rawWeights[i] = weight;
			totalWeight += weight;
		}

		if (totalWeight <= BLEND_EPSILON) {
			const nearestIndex = this._findNearestChild(parameterXValue, parameterYValue);
			return [{ clipName: this.children[nearestIndex].clipName, weight: 1 }];
		}

		return this._normalizeWeights(this._weightsFromValues(rawWeights));
	}

	private _evaluateTriangulation(
		parameterXValue: number,
		parameterYValue: number
	): BlendTreeChildWeight[] {
		if (this._triangles.length === 0) {
			return this._evaluateDirectional(parameterXValue, parameterYValue);
		}

		let bestInside: {
			triangle: BlendTree2DTriangle;
			weights: [number, number, number];
			minWeight: number;
		} | null = null;

		for (const triangle of this._triangles) {
			const weights = this._computeTriangleWeights(
				triangle.indices,
				parameterXValue,
				parameterYValue
			);
			if (!weights) continue;
			const minWeight = Math.min(weights[0], weights[1], weights[2]);
			if (minWeight < -BLEND_EPSILON) continue;
			if (!bestInside) {
				bestInside = { triangle, weights, minWeight };
				continue;
			}
			if (minWeight > bestInside.minWeight + BLEND_EPSILON) {
				bestInside = { triangle, weights, minWeight };
				continue;
			}
			if (
				Math.abs(minWeight - bestInside.minWeight) <= BLEND_EPSILON &&
				triangle.area2 < bestInside.triangle.area2
			) {
				bestInside = { triangle, weights, minWeight };
			}
		}

		if (bestInside) {
			return this._weightsFromTriangle(bestInside.triangle.indices, bestInside.weights);
		}

		let bestProjection: {
			triangle: BlendTree2DTriangle;
			weights: [number, number, number];
			distanceSquared: number;
		} | null = null;

		for (const triangle of this._triangles) {
			const weights = this._computeTriangleWeights(
				triangle.indices,
				parameterXValue,
				parameterYValue
			);
			if (!weights) continue;
			const normalized = this._clampAndNormalizeTriangleWeights(weights);
			if (!normalized) continue;
			const projected = this._projectTriangleWeights(triangle.indices, normalized);
			const dx = parameterXValue - projected[0];
			const dy = parameterYValue - projected[1];
			const distanceSquared = dx * dx + dy * dy;

			if (
				!bestProjection ||
				distanceSquared < bestProjection.distanceSquared
			) {
				bestProjection = {
					triangle,
					weights: normalized,
					distanceSquared,
				};
			}
		}

		if (bestProjection) {
			return this._weightsFromTriangle(
				bestProjection.triangle.indices,
				bestProjection.weights
			);
		}

		return this._evaluateDirectional(parameterXValue, parameterYValue);
	}

	private _computeTriangleWeights(
		indices: [number, number, number],
		pointX: number,
		pointY: number
	): [number, number, number] | null {
		const a = this.children[indices[0]];
		const b = this.children[indices[1]];
		const c = this.children[indices[2]];

		const denominator =
			(b.positionY - c.positionY) * (a.positionX - c.positionX) +
			(c.positionX - b.positionX) * (a.positionY - c.positionY);
		if (Math.abs(denominator) <= TRIANGLE_EPSILON) {
			return null;
		}

		const weightA =
			((b.positionY - c.positionY) * (pointX - c.positionX) +
				(c.positionX - b.positionX) * (pointY - c.positionY)) /
			denominator;
		const weightB =
			((c.positionY - a.positionY) * (pointX - c.positionX) +
				(a.positionX - c.positionX) * (pointY - c.positionY)) /
			denominator;
		const weightC = 1 - weightA - weightB;

		return [weightA, weightB, weightC];
	}

	private _clampAndNormalizeTriangleWeights(
		weights: [number, number, number]
	): [number, number, number] | null {
		const clampedA = Math.max(0, weights[0]);
		const clampedB = Math.max(0, weights[1]);
		const clampedC = Math.max(0, weights[2]);
		const total = clampedA + clampedB + clampedC;
		if (total <= BLEND_EPSILON) {
			return null;
		}
		return [clampedA / total, clampedB / total, clampedC / total];
	}

	private _projectTriangleWeights(
		indices: [number, number, number],
		weights: [number, number, number]
	): [number, number] {
		const a = this.children[indices[0]];
		const b = this.children[indices[1]];
		const c = this.children[indices[2]];
		return [
			a.positionX * weights[0] + b.positionX * weights[1] + c.positionX * weights[2],
			a.positionY * weights[0] + b.positionY * weights[1] + c.positionY * weights[2],
		];
	}

	private _weightsFromTriangle(
		indices: [number, number, number],
		weights: [number, number, number]
	): BlendTreeChildWeight[] {
		const normalized = this._clampAndNormalizeTriangleWeights(weights);
		if (!normalized) return [];
		const rawWeights = new Array<number>(this.children.length).fill(0);
		rawWeights[indices[0]] = normalized[0];
		rawWeights[indices[1]] = normalized[1];
		rawWeights[indices[2]] = normalized[2];
		return this._normalizeWeights(this._weightsFromValues(rawWeights));
	}

	private _weightsFromValues(values: number[]): BlendTreeChildWeight[] {
		const accumulated = new Map<string, number>();
		for (let i = 0; i < values.length; i++) {
			const value = values[i];
			if (value <= BLEND_EPSILON) continue;
			const clipName = this.children[i].clipName;
			accumulated.set(clipName, (accumulated.get(clipName) ?? 0) + value);
		}

		const weights: BlendTreeChildWeight[] = [];
		for (const child of this.children) {
			const clipWeight = accumulated.get(child.clipName);
			if (clipWeight === undefined || clipWeight <= BLEND_EPSILON) continue;
			weights.push({
				clipName: child.clipName,
				weight: clipWeight,
			});
			accumulated.delete(child.clipName);
		}
		return weights;
	}

	private _normalizeWeights(
		weights: BlendTreeChildWeight[]
	): BlendTreeChildWeight[] {
		let totalWeight = 0;
		for (const weight of weights) {
			weight.weight = Math.max(0, weight.weight);
			totalWeight += weight.weight;
		}
		if (totalWeight <= BLEND_EPSILON) {
			return [];
		}
		for (const weight of weights) {
			weight.weight /= totalWeight;
		}
		return weights.filter((weight) => weight.weight > BLEND_EPSILON);
	}

	private _findExactMatch(pointX: number, pointY: number): number {
		for (let i = 0; i < this.children.length; i++) {
			const child = this.children[i];
			const dx = pointX - child.positionX;
			const dy = pointY - child.positionY;
			const distanceSquared = dx * dx + dy * dy;
			if (distanceSquared <= TRIANGLE_EPSILON) {
				return i;
			}
		}
		return -1;
	}

	private _findNearestChild(pointX: number, pointY: number): number {
		let nearestIndex = 0;
		let nearestDistanceSquared = Number.POSITIVE_INFINITY;
		for (let i = 0; i < this.children.length; i++) {
			const child = this.children[i];
			const dx = pointX - child.positionX;
			const dy = pointY - child.positionY;
			const distanceSquared = dx * dx + dy * dy;
			if (distanceSquared < nearestDistanceSquared) {
				nearestDistanceSquared = distanceSquared;
				nearestIndex = i;
			}
		}
		return nearestIndex;
	}

	private _maxChildRadius(): number {
		let maxRadius = 0;
		for (const child of this.children) {
			const radius = Math.hypot(child.positionX, child.positionY);
			if (radius > maxRadius) {
				maxRadius = radius;
			}
		}
		return maxRadius;
	}

	private _buildTriangles(): BlendTree2DTriangle[] {
		const triangles: BlendTree2DTriangle[] = [];
		for (let i = 0; i < this.children.length - 2; i++) {
			for (let j = i + 1; j < this.children.length - 1; j++) {
				for (let k = j + 1; k < this.children.length; k++) {
					const area2 = Math.abs(
						(this.children[j].positionX - this.children[i].positionX) *
							(this.children[k].positionY - this.children[i].positionY) -
							(this.children[j].positionY - this.children[i].positionY) *
								(this.children[k].positionX - this.children[i].positionX)
					);
					if (area2 <= TRIANGLE_EPSILON) continue;
					triangles.push({
						indices: [i, j, k],
						area2,
					});
				}
			}
		}
		return triangles;
	}
}
