import type { IPrimitiveGeometry } from "../../core/types";
import type { Skeleton } from "../../animation/Skeleton";
import type { DeformedGeometryOverride } from "./types";

export interface PrimitiveDeformOptions {
	geometry: IPrimitiveGeometry;
	morphWeights?: ArrayLike<number> | null;
	skeleton?: Skeleton | null;
}

export function deformPrimitiveGeometry(
	options: PrimitiveDeformOptions
): DeformedGeometryOverride {
	const geometry = options.geometry;
	const vertexCount = (geometry.positions.length / 3) | 0;
	const positions = new Float32Array(geometry.positions);
	const normals = geometry.normals
		? new Float32Array(geometry.normals)
		: undefined;
	const tangents = geometry.tangents
		? new Float32Array(geometry.tangents)
		: undefined;

	applyMorphTargets(
		geometry,
		positions,
		normals,
		tangents,
		options.morphWeights ?? null
	);

	if (options.skeleton && geometry.joints0 && geometry.weights0) {
		applySkinning(
			positions,
			normals,
			tangents,
			geometry,
			options.skeleton,
			vertexCount
		);
	}

	return {
		positions,
		normals,
		tangents,
	};
}

function applyMorphTargets(
	geometry: IPrimitiveGeometry,
	positions: Float32Array,
	normals: Float32Array | undefined,
	tangents: Float32Array | undefined,
	weights: ArrayLike<number> | null
): void {
	const targets = geometry.morphTargets;
	if (!targets || targets.length === 0 || !weights) return;

	const count = Math.min(targets.length, weights.length);
	for (let targetIndex = 0; targetIndex < count; targetIndex++) {
		const weight = Number(weights[targetIndex] ?? 0);
		if (!Number.isFinite(weight) || Math.abs(weight) <= 1e-6) continue;
		const target = targets[targetIndex];
		if (target.positions) {
			for (let i = 0; i < positions.length; i++) {
				positions[i] += target.positions[i] * weight;
			}
		}
		if (normals && target.normals) {
			for (let i = 0; i < normals.length; i++) {
				normals[i] += target.normals[i] * weight;
			}
		}
		if (tangents && target.tangents) {
			for (let i = 0; i < tangents.length; i++) {
				tangents[i] += target.tangents[i] * weight;
			}
		}
	}
}

function applySkinning(
	positions: Float32Array,
	normals: Float32Array | undefined,
	tangents: Float32Array | undefined,
	geometry: IPrimitiveGeometry,
	skeleton: Skeleton,
	vertexCount: number
): void {
	const joints0 = geometry.joints0!;
	const weights0 = geometry.weights0!;
	const joints1 = geometry.joints1;
	const weights1 = geometry.weights1;

	skeleton.updateJointMatrices();
	const jointMatrices = skeleton.jointMatrices;

	for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
		const basePos = vertexIndex * 3;
		const px = positions[basePos];
		const py = positions[basePos + 1];
		const pz = positions[basePos + 2];

		let sx = 0;
		let sy = 0;
		let sz = 0;

		let nx = 0;
		let ny = 0;
		let nz = 0;

		let tx = 0;
		let ty = 0;
		let tz = 0;
		let tw = tangents ? tangents[vertexIndex * 4 + 3] : 1;

		let totalWeight = 0;

		for (let influence = 0; influence < 4; influence++) {
			const jointIndex = joints0[vertexIndex * 4 + influence] ?? 0;
			const weight = weights0[vertexIndex * 4 + influence] ?? 0;
			if (weight <= 1e-6) continue;
			if (jointIndex < 0 || jointIndex >= jointMatrices.length) continue;
			totalWeight += weight;
			const matrix = jointMatrices[jointIndex].elements;
			const transformed = transformPoint(matrix, px, py, pz);
			sx += transformed[0] * weight;
			sy += transformed[1] * weight;
			sz += transformed[2] * weight;

			if (normals) {
				const nBase = basePos;
				const n = transformDirection(
					matrix,
					normals[nBase],
					normals[nBase + 1],
					normals[nBase + 2]
				);
				nx += n[0] * weight;
				ny += n[1] * weight;
				nz += n[2] * weight;
			}

			if (tangents) {
				const tBase = vertexIndex * 4;
				const t = transformDirection(
					matrix,
					tangents[tBase],
					tangents[tBase + 1],
					tangents[tBase + 2]
				);
				tx += t[0] * weight;
				ty += t[1] * weight;
				tz += t[2] * weight;
			}
		}

		if (joints1 && weights1) {
			for (let influence = 0; influence < 4; influence++) {
				const jointIndex = joints1[vertexIndex * 4 + influence] ?? 0;
				const weight = weights1[vertexIndex * 4 + influence] ?? 0;
				if (weight <= 1e-6) continue;
				if (jointIndex < 0 || jointIndex >= jointMatrices.length) continue;
				totalWeight += weight;
				const matrix = jointMatrices[jointIndex].elements;
				const transformed = transformPoint(matrix, px, py, pz);
				sx += transformed[0] * weight;
				sy += transformed[1] * weight;
				sz += transformed[2] * weight;

				if (normals) {
					const nBase = basePos;
					const n = transformDirection(
						matrix,
						normals[nBase],
						normals[nBase + 1],
						normals[nBase + 2]
					);
					nx += n[0] * weight;
					ny += n[1] * weight;
					nz += n[2] * weight;
				}

				if (tangents) {
					const tBase = vertexIndex * 4;
					const t = transformDirection(
						matrix,
						tangents[tBase],
						tangents[tBase + 1],
						tangents[tBase + 2]
					);
					tx += t[0] * weight;
					ty += t[1] * weight;
					tz += t[2] * weight;
				}
			}
		}

		if (totalWeight <= 1e-6) continue;

		const inv = 1 / totalWeight;
		positions[basePos] = sx * inv;
		positions[basePos + 1] = sy * inv;
		positions[basePos + 2] = sz * inv;

		if (normals) {
			const len = Math.hypot(nx, ny, nz) || 1;
			normals[basePos] = nx / len;
			normals[basePos + 1] = ny / len;
			normals[basePos + 2] = nz / len;
		}
		if (tangents) {
			const tBase = vertexIndex * 4;
			const len = Math.hypot(tx, ty, tz) || 1;
			tangents[tBase] = tx / len;
			tangents[tBase + 1] = ty / len;
			tangents[tBase + 2] = tz / len;
			tangents[tBase + 3] = tw;
		}
	}
}

function transformPoint(
	matrix: number[][],
	x: number,
	y: number,
	z: number
): [number, number, number] {
	return [
		matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
		matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
		matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
	];
}

function transformDirection(
	matrix: number[][],
	x: number,
	y: number,
	z: number
): [number, number, number] {
	return [
		matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
		matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
		matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
	];
}
