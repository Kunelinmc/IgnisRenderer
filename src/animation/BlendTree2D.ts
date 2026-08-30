import type { Vec3Tuple } from "../maths/Vector3";
import type { BlendTreeChildWeight } from "./types";

const BLEND_EPSILON = 1e-6;
const TRIANGLE_EPSILON = 1e-8;
const CIRCUMCIRCLE_EPSILON = 1e-10;
const EDGE_CONTINUITY_SOFT_BIAS = 0.35;

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
	temporalSmoothingSeconds?: number;
	edgeContinuityBlendWidth?: number;
}

export interface BlendTree2DEvaluateOptions {
	deltaTimeSeconds?: number;
	temporalSmoothingSeconds?: number;
	edgeContinuityBlendWidth?: number;
	resetTemporalSmoothing?: boolean;
}

interface BlendTree2DTriangle {
	indices: Vec3Tuple;
	area2: number;
	neighbors: [number | null, number | null, number | null];
}

interface BlendTree2DPoint {
	x: number;
	y: number;
	childIndex: number;
}

interface DelaunayTriangle {
	vertices: Vec3Tuple;
}

interface TriangleLocation {
	triangleIndex: number;
	weights: Vec3Tuple;
}

export class BlendTree2D {
	public readonly name: string;
	public readonly parameterX: string;
	public readonly parameterY: string;
	public readonly children: BlendTree2DChild[];
	public readonly blendMode: BlendTree2DBlendMode;
	public readonly temporalSmoothingSeconds: number;
	public readonly edgeContinuityBlendWidth: number;

	private readonly _triangles: BlendTree2DTriangle[];

	private _lastTriangleIndex: number;
	private _smoothedChildWeights: number[] | null;

	constructor(options: BlendTree2DOptions) {
		this.name = options.name;
		this.parameterX = options.parameterX;
		this.parameterY = options.parameterY;
		this.children = [...options.children];
		this.blendMode = options.blendMode ?? "directional";
		this.temporalSmoothingSeconds = Math.max(
			0,
			options.temporalSmoothingSeconds ?? 0
		);
		this.edgeContinuityBlendWidth = Math.max(
			0,
			options.edgeContinuityBlendWidth ?? 0
		);
		this._triangles = this._buildTriangles();
		this._lastTriangleIndex = -1;
		this._smoothedChildWeights = null;
	}

	public evaluate(
		parameterXValue: number,
		parameterYValue: number,
		blendModeOrOptions:
			| BlendTree2DBlendMode
			| BlendTree2DEvaluateOptions = this.blendMode,
		evaluateOptions: BlendTree2DEvaluateOptions = {}
	): BlendTreeChildWeight[] {
		let blendMode = this.blendMode;
		let options = evaluateOptions;
		if (typeof blendModeOrOptions === "string") {
			blendMode = blendModeOrOptions;
		} else {
			options = blendModeOrOptions;
		}

		if (options.resetTemporalSmoothing) {
			this.resetTemporalSmoothing();
		}

		const rawWeights = this._evaluateRawWeights(
			parameterXValue,
			parameterYValue,
			blendMode,
			options.edgeContinuityBlendWidth ?? this.edgeContinuityBlendWidth
		);
		const smoothedWeights = this._applyTemporalSmoothing(
			rawWeights,
			options.deltaTimeSeconds,
			options.temporalSmoothingSeconds ?? this.temporalSmoothingSeconds
		);
		return this._normalizeWeights(this._weightsFromValues(smoothedWeights));
	}

	public resetTemporalSmoothing(): void {
		this._smoothedChildWeights = null;
		this._lastTriangleIndex = -1;
	}

	private _evaluateRawWeights(
		parameterXValue: number,
		parameterYValue: number,
		blendMode: BlendTree2DBlendMode,
		edgeContinuityBlendWidth: number
	): number[] {
		if (this.children.length === 0) return [];
		if (this.children.length === 1) return [1];

		const exactMatch = this._findExactMatch(parameterXValue, parameterYValue);
		if (exactMatch >= 0) {
			const raw = new Array<number>(this.children.length).fill(0);
			raw[exactMatch] = 1;
			return raw;
		}

		if (blendMode === "triangulation") {
			return this._evaluateTriangulationRaw(
				parameterXValue,
				parameterYValue,
				Math.max(0, edgeContinuityBlendWidth)
			);
		}
		return this._evaluateDirectionalRaw(parameterXValue, parameterYValue);
	}

	private _evaluateDirectionalRaw(
		parameterXValue: number,
		parameterYValue: number
	): number[] {
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
			const fallback = new Array<number>(this.children.length).fill(0);
			fallback[nearestIndex] = 1;
			return fallback;
		}

		return rawWeights;
	}

	private _evaluateTriangulationRaw(
		parameterXValue: number,
		parameterYValue: number,
		edgeContinuityBlendWidth: number
	): number[] {
		if (this._triangles.length === 0) {
			return this._evaluateDirectionalRaw(parameterXValue, parameterYValue);
		}

		const location = this._locateTriangle(parameterXValue, parameterYValue);
		if (location) {
			const normalized = this._clampAndNormalizeTriangleWeights(location.weights);
			if (normalized) {
				this._lastTriangleIndex = location.triangleIndex;
				let raw = this._weightsFromTriangleValues(
					this._triangles[location.triangleIndex].indices,
					normalized
				);
				if (edgeContinuityBlendWidth > BLEND_EPSILON) {
					raw = this._applyEdgeContinuityBlend(
						parameterXValue,
						parameterYValue,
						location.triangleIndex,
						raw,
						edgeContinuityBlendWidth
					);
				}
				return raw;
			}
		}

		const projection = this._projectToNearestTriangle(
			parameterXValue,
			parameterYValue
		);
		if (projection) {
			this._lastTriangleIndex = projection.triangleIndex;
			let raw = this._weightsFromTriangleValues(
				this._triangles[projection.triangleIndex].indices,
				projection.weights
			);
			if (edgeContinuityBlendWidth > BLEND_EPSILON) {
				raw = this._applyEdgeContinuityBlend(
					parameterXValue,
					parameterYValue,
					projection.triangleIndex,
					raw,
					edgeContinuityBlendWidth
				);
			}
			return raw;
		}

		return this._evaluateDirectionalRaw(parameterXValue, parameterYValue);
	}

	private _applyEdgeContinuityBlend(
		pointX: number,
		pointY: number,
		triangleIndex: number,
		baseWeights: number[],
		blendWidth: number
	): number[] {
		const triangle = this._triangles[triangleIndex];
		const blended = baseWeights.slice();
		const safeBlendWidth = Math.max(blendWidth, BLEND_EPSILON);

		for (let edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
			const neighborIndex = triangle.neighbors[edgeIndex];
			if (neighborIndex === null || neighborIndex < 0) continue;

			const edge = this._getTriangleEdgeVertices(triangle.indices, edgeIndex);
			const edgeA = this.children[edge[0]];
			const edgeB = this.children[edge[1]];
			const distance = this._distanceToLineSegment(
				pointX,
				pointY,
				edgeA.positionX,
				edgeA.positionY,
				edgeB.positionX,
				edgeB.positionY
			);
			if (distance > safeBlendWidth) continue;

			const influence = 1 - this._smoothStep(0, safeBlendWidth, distance);
			if (influence <= BLEND_EPSILON) continue;

			const neighborTriangle = this._triangles[neighborIndex];
			const neighborWeights = this._computeTriangleWeights(
				neighborTriangle.indices,
				pointX,
				pointY
			);
			if (!neighborWeights) continue;

			const softenedNeighbor = this._softClampAndNormalizeTriangleWeights(
				neighborWeights,
				influence * EDGE_CONTINUITY_SOFT_BIAS
			);
			if (!softenedNeighbor) continue;

			const neighborRaw = this._weightsFromTriangleValues(
				neighborTriangle.indices,
				softenedNeighbor
			);
			const mixAmount = 0.5 * influence;
			const keepAmount = 1 - mixAmount;
			for (let i = 0; i < blended.length; i++) {
				blended[i] = blended[i] * keepAmount + neighborRaw[i] * mixAmount;
			}
		}

		return this._normalizeRawValues(blended);
	}

	private _locateTriangle(pointX: number, pointY: number): TriangleLocation | null {
		if (this._triangles.length === 0) return null;

		let current =
			this._lastTriangleIndex >= 0 &&
			this._lastTriangleIndex < this._triangles.length
				? this._lastTriangleIndex
				: 0;
		const visited = new Set<number>();

		for (let step = 0; step < this._triangles.length + 1; step++) {
			if (visited.has(current)) break;
			visited.add(current);

			const triangle = this._triangles[current];
			const weights = this._computeTriangleWeights(
				triangle.indices,
				pointX,
				pointY
			);
			if (weights) {
				const minWeight = Math.min(weights[0], weights[1], weights[2]);
				if (minWeight >= -BLEND_EPSILON) {
					return { triangleIndex: current, weights };
				}
				let exitEdgeIndex = 0;
				let lowestWeight = weights[0];
				for (let i = 1; i < 3; i++) {
					if (weights[i] < lowestWeight) {
						lowestWeight = weights[i];
						exitEdgeIndex = i;
					}
				}
				const neighborIndex = triangle.neighbors[exitEdgeIndex];
				if (neighborIndex === null || neighborIndex < 0) break;
				current = neighborIndex;
				continue;
			}

			break;
		}

		let bestInside: {
			triangleIndex: number;
			weights: Vec3Tuple;
			minWeight: number;
		} | null = null;
		for (let i = 0; i < this._triangles.length; i++) {
			const triangle = this._triangles[i];
			const weights = this._computeTriangleWeights(
				triangle.indices,
				pointX,
				pointY
			);
			if (!weights) continue;
			const minWeight = Math.min(weights[0], weights[1], weights[2]);
			if (minWeight < -BLEND_EPSILON) continue;
			if (
				!bestInside ||
				minWeight > bestInside.minWeight + BLEND_EPSILON ||
				(Math.abs(minWeight - bestInside.minWeight) <= BLEND_EPSILON &&
					triangle.area2 < this._triangles[bestInside.triangleIndex].area2)
			) {
				bestInside = {
					triangleIndex: i,
					weights,
					minWeight,
				};
			}
		}

		if (bestInside) {
			return {
				triangleIndex: bestInside.triangleIndex,
				weights: bestInside.weights,
			};
		}
		return null;
	}

	private _projectToNearestTriangle(
		pointX: number,
		pointY: number
	): { triangleIndex: number; weights: Vec3Tuple } | null {
		let bestProjection: {
			triangleIndex: number;
			weights: Vec3Tuple;
			distanceSquared: number;
		} | null = null;

		for (let i = 0; i < this._triangles.length; i++) {
			const triangle = this._triangles[i];
			const weights = this._computeTriangleWeights(triangle.indices, pointX, pointY);
			if (!weights) continue;
			const normalized = this._clampAndNormalizeTriangleWeights(weights);
			if (!normalized) continue;
			const projected = this._projectTriangleWeights(triangle.indices, normalized);
			const dx = pointX - projected[0];
			const dy = pointY - projected[1];
			const distanceSquared = dx * dx + dy * dy;

			if (!bestProjection || distanceSquared < bestProjection.distanceSquared) {
				bestProjection = {
					triangleIndex: i,
					weights: normalized,
					distanceSquared,
				};
			}
		}

		if (!bestProjection) return null;
		return {
			triangleIndex: bestProjection.triangleIndex,
			weights: bestProjection.weights,
		};
	}

	private _applyTemporalSmoothing(
		rawWeights: number[],
		deltaTimeSeconds: number | undefined,
		smoothingSeconds: number
	): number[] {
		if (rawWeights.length === 0) {
			this._smoothedChildWeights = null;
			return rawWeights;
		}
		const safeSmoothing = Math.max(0, smoothingSeconds);
		if (safeSmoothing <= BLEND_EPSILON) {
			this._smoothedChildWeights = rawWeights.slice();
			return rawWeights;
		}

		if (
			!this._smoothedChildWeights ||
			this._smoothedChildWeights.length !== rawWeights.length
		) {
			this._smoothedChildWeights = rawWeights.slice();
			return rawWeights;
		}

		const dt = Math.max(0, deltaTimeSeconds ?? 0);
		if (dt <= BLEND_EPSILON) {
			this._smoothedChildWeights = rawWeights.slice();
			return rawWeights;
		}

		const alpha = 1 - Math.exp(-dt / safeSmoothing);
		const smoothed = new Array<number>(rawWeights.length);
		for (let i = 0; i < rawWeights.length; i++) {
			const previous = this._smoothedChildWeights[i];
			smoothed[i] = previous + (rawWeights[i] - previous) * alpha;
		}
		this._smoothedChildWeights = smoothed;
		return smoothed;
	}

	private _computeTriangleWeights(
		indices: Vec3Tuple,
		pointX: number,
		pointY: number
	): Vec3Tuple | null {
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
		weights: Vec3Tuple
	): Vec3Tuple | null {
		const clampedA = Math.max(0, weights[0]);
		const clampedB = Math.max(0, weights[1]);
		const clampedC = Math.max(0, weights[2]);
		const total = clampedA + clampedB + clampedC;
		if (total <= BLEND_EPSILON) {
			return null;
		}
		return [clampedA / total, clampedB / total, clampedC / total];
	}

	private _softClampAndNormalizeTriangleWeights(
		weights: Vec3Tuple,
		softBias: number
	): Vec3Tuple | null {
		const bias = Math.max(0, softBias);
		const clampedA = Math.max(0, weights[0] + bias);
		const clampedB = Math.max(0, weights[1] + bias);
		const clampedC = Math.max(0, weights[2] + bias);
		const total = clampedA + clampedB + clampedC;
		if (total <= BLEND_EPSILON) {
			return null;
		}
		return [clampedA / total, clampedB / total, clampedC / total];
	}

	private _projectTriangleWeights(
		indices: Vec3Tuple,
		weights: Vec3Tuple
	): [number, number] {
		const a = this.children[indices[0]];
		const b = this.children[indices[1]];
		const c = this.children[indices[2]];
		return [
			a.positionX * weights[0] + b.positionX * weights[1] + c.positionX * weights[2],
			a.positionY * weights[0] + b.positionY * weights[1] + c.positionY * weights[2],
		];
	}

	private _weightsFromTriangleValues(
		indices: Vec3Tuple,
		weights: Vec3Tuple
	): number[] {
		const rawWeights = new Array<number>(this.children.length).fill(0);
		rawWeights[indices[0]] = weights[0];
		rawWeights[indices[1]] = weights[1];
		rawWeights[indices[2]] = weights[2];
		return rawWeights;
	}

	private _normalizeRawValues(values: number[]): number[] {
		let total = 0;
		for (let i = 0; i < values.length; i++) {
			values[i] = Math.max(0, values[i]);
			total += values[i];
		}
		if (total <= BLEND_EPSILON) return values;
		for (let i = 0; i < values.length; i++) {
			values[i] /= total;
		}
		return values;
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
		if (this.children.length < 3) return [];

		const uniquePoints = this._collectUniquePoints();
		if (uniquePoints.length < 3) return [];

		const delaunay = this._buildDelaunayTriangles(uniquePoints);
		if (delaunay.length === 0) return [];

		const triangles: BlendTree2DTriangle[] = [];
		const seen = new Set<string>();
		for (const triangle of delaunay) {
			const indices: Vec3Tuple = [
				uniquePoints[triangle.vertices[0]].childIndex,
				uniquePoints[triangle.vertices[1]].childIndex,
				uniquePoints[triangle.vertices[2]].childIndex,
			];
			const dedupeKey = [...indices].sort((left, right) => left - right).join(":");
			if (seen.has(dedupeKey)) continue;
			seen.add(dedupeKey);

			const area2 = Math.abs(this._childTriangleArea2(indices));
			if (area2 <= TRIANGLE_EPSILON) continue;
			triangles.push({
				indices,
				area2,
				neighbors: [null, null, null],
			});
		}

		this._buildTriangleAdjacency(triangles);
		return triangles;
	}

	private _collectUniquePoints(): BlendTree2DPoint[] {
		const points: BlendTree2DPoint[] = [];
		for (let i = 0; i < this.children.length; i++) {
			const child = this.children[i];
			let duplicate = false;
			for (let j = 0; j < points.length; j++) {
				const point = points[j];
				const dx = point.x - child.positionX;
				const dy = point.y - child.positionY;
				if (dx * dx + dy * dy <= TRIANGLE_EPSILON) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) continue;
			points.push({
				x: child.positionX,
				y: child.positionY,
				childIndex: i,
			});
		}
		points.sort((left, right) => {
			if (Math.abs(left.x - right.x) > TRIANGLE_EPSILON) {
				return left.x - right.x;
			}
			return left.y - right.y;
		});
		return points;
	}

	private _buildDelaunayTriangles(points: BlendTree2DPoint[]): DelaunayTriangle[] {
		let minX = Number.POSITIVE_INFINITY;
		let minY = Number.POSITIVE_INFINITY;
		let maxX = Number.NEGATIVE_INFINITY;
		let maxY = Number.NEGATIVE_INFINITY;
		for (const point of points) {
			if (point.x < minX) minX = point.x;
			if (point.y < minY) minY = point.y;
			if (point.x > maxX) maxX = point.x;
			if (point.y > maxY) maxY = point.y;
		}

		const spanX = maxX - minX;
		const spanY = maxY - minY;
		const maxSpan = Math.max(spanX, spanY, 1);
		const midX = (minX + maxX) * 0.5;
		const midY = (minY + maxY) * 0.5;

		const superA: BlendTree2DPoint = {
			x: midX - maxSpan * 20,
			y: midY - maxSpan * 20,
			childIndex: -1,
		};
		const superB: BlendTree2DPoint = {
			x: midX,
			y: midY + maxSpan * 20,
			childIndex: -1,
		};
		const superC: BlendTree2DPoint = {
			x: midX + maxSpan * 20,
			y: midY - maxSpan * 20,
			childIndex: -1,
		};

		const allPoints = points.concat([superA, superB, superC]);
		const superAIndex = allPoints.length - 3;
		const superBIndex = allPoints.length - 2;
		const superCIndex = allPoints.length - 1;

		let initial: DelaunayTriangle = {
			vertices: [superAIndex, superBIndex, superCIndex],
		};
		if (
			this._signedArea2Points(
				allPoints[initial.vertices[0]],
				allPoints[initial.vertices[1]],
				allPoints[initial.vertices[2]]
			) < 0
		) {
			initial = {
				vertices: [superAIndex, superCIndex, superBIndex],
			};
		}

		let triangles: DelaunayTriangle[] = [initial];

		for (let pointIndex = 0; pointIndex < points.length; pointIndex++) {
			const badTriangleIndices: number[] = [];
			for (let triangleIndex = 0; triangleIndex < triangles.length; triangleIndex++) {
				if (
					this._isPointInTriangleCircumcircle(
						pointIndex,
						triangles[triangleIndex],
						allPoints
					)
				) {
					badTriangleIndices.push(triangleIndex);
				}
			}

			if (badTriangleIndices.length === 0) continue;

			const edgeMap = new Map<
				string,
				{ start: number; end: number; count: number }
			>();
			for (const triangleIndex of badTriangleIndices) {
				const triangle = triangles[triangleIndex];
				const edges = [
					[triangle.vertices[0], triangle.vertices[1]],
					[triangle.vertices[1], triangle.vertices[2]],
					[triangle.vertices[2], triangle.vertices[0]],
				] as const;
				for (const edge of edges) {
					const key = this._edgeKey(edge[0], edge[1]);
					const entry = edgeMap.get(key);
					if (entry) {
						entry.count++;
					} else {
						edgeMap.set(key, {
							start: edge[0],
							end: edge[1],
							count: 1,
						});
					}
				}
			}

			const badSet = new Set<number>(badTriangleIndices);
			const kept: DelaunayTriangle[] = [];
			for (let i = 0; i < triangles.length; i++) {
				if (!badSet.has(i)) {
					kept.push(triangles[i]);
				}
			}
			triangles = kept;

			for (const entry of edgeMap.values()) {
				if (entry.count !== 1) continue;
				const oriented = this._orientTriangleVertices(
					entry.start,
					entry.end,
					pointIndex,
					allPoints
				);
				if (!oriented) continue;
				triangles.push({ vertices: oriented });
			}
		}

		const pointCount = points.length;
		return triangles.filter((triangle) => {
			return (
				triangle.vertices[0] < pointCount &&
				triangle.vertices[1] < pointCount &&
				triangle.vertices[2] < pointCount
			);
		});
	}

	private _isPointInTriangleCircumcircle(
		pointIndex: number,
		triangle: DelaunayTriangle,
		points: BlendTree2DPoint[]
	): boolean {
		const point = points[pointIndex];
		const a = points[triangle.vertices[0]];
		const b = points[triangle.vertices[1]];
		const c = points[triangle.vertices[2]];

		const ax = a.x - point.x;
		const ay = a.y - point.y;
		const bx = b.x - point.x;
		const by = b.y - point.y;
		const cx = c.x - point.x;
		const cy = c.y - point.y;

		const determinant =
			(ax * ax + ay * ay) * (bx * cy - by * cx) -
			(bx * bx + by * by) * (ax * cy - ay * cx) +
			(cx * cx + cy * cy) * (ax * by - ay * bx);
		return determinant > CIRCUMCIRCLE_EPSILON;
	}

	private _orientTriangleVertices(
		a: number,
		b: number,
		c: number,
		points: BlendTree2DPoint[]
	): Vec3Tuple | null {
		const area2 = this._signedArea2Points(points[a], points[b], points[c]);
		if (Math.abs(area2) <= TRIANGLE_EPSILON) return null;
		if (area2 > 0) {
			return [a, b, c];
		}
		return [a, c, b];
	}

	private _buildTriangleAdjacency(triangles: BlendTree2DTriangle[]): void {
		const edgeOwners = new Map<
			string,
			{ triangleIndex: number; edgeIndex: number }
		>();
		for (let triangleIndex = 0; triangleIndex < triangles.length; triangleIndex++) {
			const triangle = triangles[triangleIndex];
			for (let edgeIndex = 0; edgeIndex < 3; edgeIndex++) {
				const edge = this._getTriangleEdgeVertices(triangle.indices, edgeIndex);
				const key = this._edgeKey(edge[0], edge[1]);
				const owner = edgeOwners.get(key);
				if (!owner) {
					edgeOwners.set(key, { triangleIndex, edgeIndex });
					continue;
				}
				triangle.neighbors[edgeIndex] = owner.triangleIndex;
				triangles[owner.triangleIndex].neighbors[owner.edgeIndex] =
					triangleIndex;
			}
		}
	}

	private _getTriangleEdgeVertices(
		indices: Vec3Tuple,
		edgeIndex: number
	): [number, number] {
		switch (edgeIndex) {
			case 0:
				return [indices[1], indices[2]];
			case 1:
				return [indices[2], indices[0]];
			default:
				return [indices[0], indices[1]];
		}
	}

	private _edgeKey(a: number, b: number): string {
		return a < b ? `${a}:${b}` : `${b}:${a}`;
	}

	private _childTriangleArea2(indices: Vec3Tuple): number {
		const a = this.children[indices[0]];
		const b = this.children[indices[1]];
		const c = this.children[indices[2]];
		return (
			(b.positionX - a.positionX) * (c.positionY - a.positionY) -
			(b.positionY - a.positionY) * (c.positionX - a.positionX)
		);
	}

	private _signedArea2Points(
		a: BlendTree2DPoint,
		b: BlendTree2DPoint,
		c: BlendTree2DPoint
	): number {
		return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
	}

	private _distanceToLineSegment(
		pointX: number,
		pointY: number,
		segmentAX: number,
		segmentAY: number,
		segmentBX: number,
		segmentBY: number
	): number {
		const abX = segmentBX - segmentAX;
		const abY = segmentBY - segmentAY;
		const abLengthSquared = abX * abX + abY * abY;
		if (abLengthSquared <= TRIANGLE_EPSILON) {
			return Math.hypot(pointX - segmentAX, pointY - segmentAY);
		}

		const apX = pointX - segmentAX;
		const apY = pointY - segmentAY;
		const t = Math.max(
			0,
			Math.min(1, (apX * abX + apY * abY) / abLengthSquared)
		);
		const closestX = segmentAX + abX * t;
		const closestY = segmentAY + abY * t;
		return Math.hypot(pointX - closestX, pointY - closestY);
	}

	private _smoothStep(min: number, max: number, value: number): number {
		if (max <= min + BLEND_EPSILON) return value >= max ? 1 : 0;
		const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
		return t * t * (3 - 2 * t);
	}
}
