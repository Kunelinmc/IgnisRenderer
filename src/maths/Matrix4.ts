/**
 * Matrix4 class and utility functions (4x4 Matrix)
 */

import { Vector3 } from "./Vector3";
import type { Point, IVector3, Matrix3Arr } from "./types";

/**
 * MATRIX CONVENTIONS:
 * - Handedness: Right-Handed
 * - Projection: Standard Z-buffer range [-1, 1] in clip space (NDC)
 * - View Matrix: LookAt creates a system where -Z is forward
 */

export class Matrix4 {
	public elements: number[][];

	constructor(elements?: number[][]) {
		this.elements = elements || [
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		];
	}

	public static identity(): Matrix4 {
		return new Matrix4([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		]);
	}

	public static multiply(
		a: Matrix4 | number[][],
		b: Matrix4 | number[][],
		out?: Matrix4
	): Matrix4 {
		const ae = a instanceof Matrix4 ? a.elements : a;
		const be = b instanceof Matrix4 ? b.elements : b;

		if (out) {
			const oe = out.elements;
			// Manual unroll for 4x4 performance
			for (let i = 0; i < 4; i++) {
				const ai0 = ae[i][0],
					ai1 = ae[i][1],
					ai2 = ae[i][2],
					ai3 = ae[i][3];
				oe[i][0] =
					ai0 * be[0][0] + ai1 * be[1][0] + ai2 * be[2][0] + ai3 * be[3][0];
				oe[i][1] =
					ai0 * be[0][1] + ai1 * be[1][1] + ai2 * be[2][1] + ai3 * be[3][1];
				oe[i][2] =
					ai0 * be[0][2] + ai1 * be[1][2] + ai2 * be[2][2] + ai3 * be[3][2];
				oe[i][3] =
					ai0 * be[0][3] + ai1 * be[1][3] + ai2 * be[2][3] + ai3 * be[3][3];
			}
			return out;
		}

		const res: number[][] = Array(4)
			.fill(null)
			.map(() => Array(4).fill(0));

		for (let i = 0; i < 4; i++) {
			for (let j = 0; j < 4; j++) {
				res[i][j] =
					ae[i][0] * be[0][j] +
					ae[i][1] * be[1][j] +
					ae[i][2] * be[2][j] +
					ae[i][3] * be[3][j];
			}
		}

		return new Matrix4(res);
	}

	public multiply(other: Matrix4 | number[][]): this {
		const result = Matrix4.multiply(this, other);
		this.elements = result.elements;
		return this;
	}

	public static transformPoint(
		m: Matrix4 | number[][],
		point: Point,
		out?: Point
	): Point {
		const me = m instanceof Matrix4 ? m.elements : m;
		const x = point.x || 0;
		const y = point.y || 0;
		const z = point.z || 0;

		const rx = me[0][0] * x + me[0][1] * y + me[0][2] * z + me[0][3];
		const ry = me[1][0] * x + me[1][1] * y + me[1][2] * z + me[1][3];
		const rz = me[2][0] * x + me[2][1] * y + me[2][2] * z + me[2][3];
		const rw = me[3][0] * x + me[3][1] * y + me[3][2] * z + me[3][3];

		if (out) {
			out.x = rx;
			out.y = ry;
			out.z = rz;
			out.w = rw;
			return out;
		}

		return { x: rx, y: ry, z: rz, w: rw };
	}

	public transformPoint(point: Point): Point {
		return Matrix4.transformPoint(this, point);
	}

	public static transformDirection(
		m: Matrix4 | number[][],
		direction: IVector3
	): Vector3 {
		const me = m instanceof Matrix4 ? m.elements : m;
		const x = direction.x || 0;
		const y = direction.y || 0;
		const z = direction.z || 0;

		return new Vector3(
			me[0][0] * x + me[0][1] * y + me[0][2] * z,
			me[1][0] * x + me[1][1] * y + me[1][2] * z,
			me[2][0] * x + me[2][1] * y + me[2][2] * z
		);
	}

	public transformDirection(direction: IVector3): Vector3 {
		return Matrix4.transformDirection(this, direction);
	}

	public static rotationFromEuler(x: number, y: number, z: number): Matrix4 {
		const cosX = Math.cos(x),
			sinX = Math.sin(x);
		const cosY = Math.cos(y),
			sinY = Math.sin(y);
		const cosZ = Math.cos(z),
			sinZ = Math.sin(z);

		// Order: Z -> Y -> X
		return new Matrix4([
			[cosY * cosZ, -cosY * sinZ, sinY, 0],
			[
				cosX * sinZ + sinX * sinY * cosZ,
				cosX * cosZ - sinX * sinY * sinZ,
				-sinX * cosY,
				0,
			],
			[
				sinX * sinZ - cosX * sinY * cosZ,
				sinX * cosZ + cosX * sinY * sinZ,
				cosX * cosY,
				0,
			],
			[0, 0, 0, 1],
		]);
	}

	/**
	 * Creates a reflection matrix for a given plane.
	 * Plane equation: ax + by + cz + d = 0
	 * Reflection matrix R = I - 2 * n * n^T (with translation)
	 */
	public static reflection(plane: {
		normal: { x: number; y: number; z: number };
		constant: number;
	}): Matrix4 {
		const a = plane.normal.x;
		const b = plane.normal.y;
		const c = plane.normal.z;
		const d = plane.constant;

		return new Matrix4([
			[1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d],
			[-2 * b * a, 1 - 2 * b * b, -2 * b * c, -2 * b * d],
			[-2 * c * a, -2 * c * b, 1 - 2 * c * c, -2 * c * d],
			[0, 0, 0, 1],
		]);
	}

	public static lookAt(eye: IVector3, target: IVector3, up: IVector3): Matrix4 {
		const z = Vector3.sub(eye, target).normalize();
		const x = Vector3.cross(up, z).normalize();
		const y = Vector3.cross(z, x);

		return new Matrix4([
			[x.x, x.y, x.z, -Vector3.dot(x, eye)],
			[y.x, y.y, y.z, -Vector3.dot(y, eye)],
			[z.x, z.y, z.z, -Vector3.dot(z, eye)],
			[0, 0, 0, 1],
		]);
	}

	public static ortho(
		left: number,
		right: number,
		bottom: number,
		top: number,
		near: number,
		far: number
	): Matrix4 {
		const lr = 1 / (left - right);
		const bt = 1 / (bottom - top);
		const nf = 1 / (near - far);

		return new Matrix4([
			[-2 * lr, 0, 0, (left + right) * lr],
			[0, -2 * bt, 0, (bottom + top) * bt],
			[0, 0, 2 * nf, (near + far) * nf],
			[0, 0, 0, 1],
		]);
	}

	public static perspective(
		fov: number,
		aspect: number,
		near: number,
		far: number
	): Matrix4 {
		const f = 1.0 / Math.tan((fov * Math.PI) / 360);
		const rangeInv = 1.0 / (near - far);

		return new Matrix4([
			[f / aspect, 0, 0, 0],
			[0, f, 0, 0],
			[0, 0, (far + near) * rangeInv, 2 * far * near * rangeInv],
			[0, 0, -1, 0],
		]);
	}

	public static inverse3x3(m: Matrix4 | number[][]): Matrix3Arr | null {
		const me = m instanceof Matrix4 ? m.elements : m;
		const m00 = me[0][0],
			m01 = me[0][1],
			m02 = me[0][2];
		const m10 = me[1][0],
			m11 = me[1][1],
			m12 = me[1][2];
		const m20 = me[2][0],
			m21 = me[2][1],
			m22 = me[2][2];

		const det =
			m00 * (m11 * m22 - m12 * m21) -
			m01 * (m10 * m22 - m12 * m20) +
			m02 * (m10 * m21 - m11 * m20);

		if (Math.abs(det) < 1e-10) {
			return null;
		}

		const invDet = 1.0 / det;

		return [
			[
				(m11 * m22 - m12 * m21) * invDet,
				(m02 * m21 - m01 * m22) * invDet,
				(m01 * m12 - m02 * m11) * invDet,
			],
			[
				(m12 * m20 - m10 * m22) * invDet,
				(m00 * m22 - m02 * m20) * invDet,
				(m02 * m10 - m00 * m12) * invDet,
			],
			[
				(m10 * m21 - m11 * m20) * invDet,
				(m01 * m20 - m00 * m21) * invDet,
				(m00 * m11 - m01 * m10) * invDet,
			],
		];
	}

	public static transpose3x3(m: Matrix3Arr): Matrix3Arr {
		return [
			[m[0][0], m[1][0], m[2][0]],
			[m[0][1], m[1][1], m[2][1]],
			[m[0][2], m[1][2], m[2][2]],
		];
	}

	public static normalMatrix(
		modelMatrix: Matrix4 | number[][],
		out?: Matrix4
	): Matrix3Arr | Matrix4 {
		const inv = Matrix4.inverse3x3(modelMatrix);
		if (!inv) {
			if (out) {
				const oe = out.elements;
				oe[0][0] = 1;
				oe[0][1] = 0;
				oe[0][2] = 0;
				oe[1][0] = 0;
				oe[1][1] = 1;
				oe[1][2] = 0;
				oe[2][0] = 0;
				oe[2][1] = 0;
				oe[2][2] = 1;
				return out;
			}
			return [
				[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1],
			];
		}
		const res = Matrix4.transpose3x3(inv);
		if (out) {
			const oe = out.elements;
			oe[0][0] = res[0][0];
			oe[0][1] = res[0][1];
			oe[0][2] = res[0][2];
			oe[1][0] = res[1][0];
			oe[1][1] = res[1][1];
			oe[1][2] = res[1][2];
			oe[2][0] = res[2][0];
			oe[2][1] = res[2][1];
			oe[2][2] = res[2][2];
			return out;
		}
		return res;
	}

	public static transformNormal(
		normalMat3x3: Matrix3Arr | Matrix4,
		direction: IVector3,
		out?: Vector3
	): Vector3 {
		const me =
			normalMat3x3 instanceof Matrix4 ? normalMat3x3.elements : normalMat3x3;
		const x = direction.x || 0;
		const y = direction.y || 0;
		const z = direction.z || 0;

		const rx = me[0][0] * x + me[0][1] * y + me[0][2] * z;
		const ry = me[1][0] * x + me[1][1] * y + me[1][2] * z;
		const rz = me[2][0] * x + me[2][1] * y + me[2][2] * z;

		if (out) {
			out.x = rx;
			out.y = ry;
			out.z = rz;
			return out;
		}

		return new Vector3(rx, ry, rz);
	}

	public static fromArray(arr: number[]): Matrix4 {
		return new Matrix4([
			[arr[0], arr[4], arr[8], arr[12]],
			[arr[1], arr[5], arr[9], arr[13]],
			[arr[2], arr[6], arr[10], arr[14]],
			[arr[3], arr[7], arr[11], arr[15]],
		]);
	}

	public static fromTranslation(t: number[]): Matrix4 {
		const m = Matrix4.identity();
		m.elements[0][3] = t[0];
		m.elements[1][3] = t[1];
		m.elements[2][3] = t[2];
		return m;
	}

	public static fromScale(s: number[]): Matrix4 {
		const m = Matrix4.identity();
		m.elements[0][0] = s[0];
		m.elements[1][1] = s[1];
		m.elements[2][2] = s[2];
		return m;
	}

	public static fromQuaternion(q: number[]): Matrix4 {
		const x = q[0],
			y = q[1],
			z = q[2],
			w = q[3];
		const x2 = x + x,
			y2 = y + y,
			z2 = z + z;
		const xx = x * x2,
			xy = x * y2,
			xz = x * z2;
		const yy = y * y2,
			yz = y * z2,
			zz = z * z2;
		const wx = w * x2,
			wy = w * y2,
			wz = w * z2;

		const m = Matrix4.identity();
		m.elements[0][0] = 1 - (yy + zz);
		m.elements[0][1] = xy - wz;
		m.elements[0][2] = xz + wy;

		m.elements[1][0] = xy + wz;
		m.elements[1][1] = 1 - (xx + zz);
		m.elements[1][2] = yz - wx;

		m.elements[2][0] = xz - wy;
		m.elements[2][1] = yz + wx;
		m.elements[2][2] = 1 - (xx + yy);

		return m;
	}

	/**
	 * Creates a deep copy of this matrix.
	 */
	public clone(): Matrix4 {
		return new Matrix4(this.elements.map((row) => [...row]));
	}

	/**
	 * Modifies this projection matrix to use an arbitrary clipping plane as its near plane.
	 * The plane should be in camera space.
	 * Based on Eric Lengyel's method for Right-Handed coordinates with NDC range [-1, 1].
	 * @param plane The clipping plane in camera space.
	 */
	public applyObliqueClipping(plane: {
		normal: { x: number; y: number; z: number };
		constant: number;
	}): void {
		const m = this.elements;
		const n = plane.normal;
		const c = plane.constant;

		// Calculate the clip-space corner point opposite the clipping plane
		// and transform it into camera space.
		// For our perspective matrix (row-major):
		// Row 0: [f/aspect, 0, 0, 0]
		// Row 1: [0, f, 0, 0]
		// Row 2: [0, 0, (F+N)/(N-F), 2FN/(N-F)]
		// Row 3: [0, 0, -1, 0]
		const qx = (Math.sign(n.x) + m[0][2]) / m[0][0];
		const qy = (Math.sign(n.y) + m[1][2]) / m[1][1];
		const qz = -1.0;
		const qw = (1.0 + m[2][2]) / m[2][3];

		// Calculate the scaled plane vector: c = C * (2.0 / (C dot Q))
		const dot = n.x * qx + n.y * qy + n.z * qz + c * qw;
		const scale = 2.0 / dot;

		const cx = n.x * scale;
		const cy = n.y * scale;
		const cz = n.z * scale;
		const cw = c * scale;

		// Replace the third row (index 2) of the projection matrix.
		// M'[2] = scaledPlane - M[3]
		// Since M[3] = [0, 0, -1, 0], we have:
		m[2][0] = cx;
		m[2][1] = cy;
		m[2][2] = cz + 1.0;
		m[2][3] = cw;
	}
}
