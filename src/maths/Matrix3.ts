/**
 * Matrix3 class and utility functions (3x3 Matrix)
 */

import { Vector3 } from "./Vector3";
import type { IVector3, IVector4 } from "./types";

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

	/**
	 * Builds a row-major rotation matrix from a quaternion.
	 *
	 * @param quaternion - Quaternion source as `{ x, y, z, w }` or
	 * `[x, y, z, w]`.
	 * @param out - Optional matrix that receives the converted rotation.
	 * @returns The converted 3x3 rotation matrix.
	 * @constraints `quaternion` should be normalized before conversion.
	 * @sideEffects Writes to `out` when provided.
	 */
	public static fromQuaternion(
		quaternion: IVector4 | ArrayLike<number>,
		out?: Matrix3
	): Matrix3 {
		const source = quaternion as Partial<IVector4> & ArrayLike<number>;
		const x = source.x ?? source[0];
		const y = source.y ?? source[1];
		const z = source.z ?? source[2];
		const w = source.w ?? source[3];
		const x2 = x + x;
		const y2 = y + y;
		const z2 = z + z;
		const xx = x * x2;
		const xy = x * y2;
		const xz = x * z2;
		const yy = y * y2;
		const yz = y * z2;
		const zz = z * z2;
		const wx = w * x2;
		const wy = w * y2;
		const wz = w * z2;
		const target = out ?? Matrix3.identity();
		const elements = target.elements;

		elements[0][0] = 1 - (yy + zz);
		elements[0][1] = xy - wz;
		elements[0][2] = xz + wy;
		elements[1][0] = xy + wz;
		elements[1][1] = 1 - (xx + zz);
		elements[1][2] = yz - wx;
		elements[2][0] = xz - wy;
		elements[2][1] = yz + wx;
		elements[2][2] = 1 - (xx + yy);

		return target;
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

	/**
	 * Copies another 3x3 matrix into this matrix.
	 *
	 * @param source - Matrix source in the engine row-major representation.
	 * @returns This matrix after the copy.
	 * @constraints `source` must contain at least three rows and three columns.
	 * @sideEffects Mutates this matrix's existing `elements` rows.
	 */
	public copy(source: Matrix3 | number[][]): this {
		const sourceElements = source instanceof Matrix3 ? source.elements : source;
		const targetElements = this.elements;
		for (let row = 0; row < 3; row++) {
			targetElements[row][0] = sourceElements[row][0];
			targetElements[row][1] = sourceElements[row][1];
			targetElements[row][2] = sourceElements[row][2];
		}
		return this;
	}

	/**
	 * Copies this 3x3 matrix into the provided target matrix.
	 *
	 * @param target - Matrix instance that receives this matrix's values.
	 * @returns The target matrix after the copy.
	 * @constraints `target` must be a writable `Matrix3` instance.
	 * @sideEffects Mutates `target.elements`.
	 */
	public copyTo(target: Matrix3): Matrix3 {
		return target.copy(this);
	}

	/**
	 * Creates a deep copy of this 3x3 matrix.
	 *
	 * @returns A new `Matrix3` with independent row arrays.
	 * @constraints None.
	 * @sideEffects None.
	 */
	public clone(): Matrix3 {
		return new Matrix3(this.elements.map((row) => [...row]));
	}
}
