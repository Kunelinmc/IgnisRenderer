/**
 * Matrix3 class and utility functions (3x3 Matrix)
 */

import { Vector3 } from "./Vector3";
import type { IVector3 } from "./types";

export class Matrix3 {
	public elements: number[][];

	constructor(elements?: number[][]) {
		this.elements = elements || [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
		];
	}

	public static identity(): Matrix3 {
		return new Matrix3([
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
		]);
	}

	public static multiply(
		a: Matrix3 | number[][],
		b: Matrix3 | number[][]
	): Matrix3 {
		const ae = a instanceof Matrix3 ? a.elements : a;
		const be = b instanceof Matrix3 ? b.elements : b;

		const res: number[][] = Array(3)
			.fill(null)
			.map(() => Array(3).fill(0));

		for (let i = 0; i < 3; i++) {
			for (let j = 0; j < 3; j++) {
				res[i][j] =
					ae[i][0] * be[0][j] + ae[i][1] * be[1][j] + ae[i][2] * be[2][j];
			}
		}

		return new Matrix3(res);
	}

	public multiply(other: Matrix3 | number[][]): this {
		const result = Matrix3.multiply(this, other);
		this.elements = result.elements;
		return this;
	}

	public static transformVector(m: Matrix3 | number[][], v: IVector3): Vector3 {
		const me = m instanceof Matrix3 ? m.elements : m;
		const x = v.x || 0;
		const y = v.y || 0;
		const z = v.z || 0;

		return new Vector3(
			me[0][0] * x + me[0][1] * y + me[0][2] * z,
			me[1][0] * x + me[1][1] * y + me[1][2] * z,
			me[2][0] * x + me[2][1] * y + me[2][2] * z
		);
	}

	public transformVector(v: IVector3): Vector3 {
		return Matrix3.transformVector(this, v);
	}

	public static fromArray(arr: number[]): Matrix3 {
		return new Matrix3([
			[arr[0], arr[3], arr[6]],
			[arr[1], arr[4], arr[7]],
			[arr[2], arr[5], arr[8]],
		]);
	}

	public static transpose(m: Matrix3 | number[][]): Matrix3 {
		const me = m instanceof Matrix3 ? m.elements : m;
		return new Matrix3([
			[me[0][0], me[1][0], me[2][0]],
			[me[0][1], me[1][1], me[2][1]],
			[me[0][2], me[1][2], me[2][2]],
		]);
	}

	public transpose(): this {
		const result = Matrix3.transpose(this);
		this.elements = result.elements;
		return this;
	}
}
