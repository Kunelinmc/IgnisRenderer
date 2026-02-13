import type { IVector3 } from "./types";

export class Quaternion {
	x: number;
	y: number;
	z: number;
	w: number;

	constructor(x = 0, y = 0, z = 0, w = 1) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}

	public static fromEuler(x: number, y: number, z: number): Quaternion {
		const c1 = Math.cos(x / 2);
		const c2 = Math.cos(y / 2);
		const c3 = Math.cos(z / 2);
		const s1 = Math.sin(x / 2);
		const s2 = Math.sin(y / 2);
		const s3 = Math.sin(z / 2);

		return new Quaternion(
			s1 * c2 * c3 + c1 * s2 * s3,
			c1 * s2 * c3 - s1 * c2 * s3,
			c1 * c2 * s3 + s1 * s2 * c3,
			c1 * c2 * c3 - s1 * s2 * s3
		);
	}

	public static fromAxisAngle(axis: IVector3, angle: number): Quaternion {
		const halfAngle = angle / 2;
		const s = Math.sin(halfAngle);
		return new Quaternion(
			axis.x * s,
			axis.y * s,
			axis.z * s,
			Math.cos(halfAngle)
		);
	}

	public static multiply(q1: Quaternion, q2: Quaternion): Quaternion {
		return new Quaternion(
			q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
			q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
			q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
			q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
		);
	}

	/**
	 * 四元数球面线性插值 (Spherical Linear Interpolation)
	 * @param {Quaternion} q1 起始四元数
	 * @param {Quaternion} q2 目标四元数
	 * @param {number} t 插值因子 [0, 1]
	 * @returns {Quaternion} 插值结果
	 */
	public static slerp(q1: Quaternion, q2: Quaternion, t: number): Quaternion {
		// 确保 t 在 [0, 1] 范围内
		t = Math.max(0, Math.min(1, t));

		// 计算两个四元数的点积
		let dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;

		// 如果点积为负，反转其中一个四元数以保证最短路径插值
		let q2Adjusted = q2;
		if (dot < 0) {
			dot = -dot;
			q2Adjusted = new Quaternion(-q2.x, -q2.y, -q2.z, -q2.w);
		}

		// 如果两个四元数非常接近，使用线性插值避免数值问题
		if (dot > 0.9995) {
			const result = new Quaternion(
				q1.x + t * (q2Adjusted.x - q1.x),
				q1.y + t * (q2Adjusted.y - q1.y),
				q1.z + t * (q2Adjusted.z - q1.z),
				q1.w + t * (q2Adjusted.w - q1.w)
			);
			return result.normalize();
		}

		// 计算插值参数
		const theta0 = Math.acos(dot); // 两个四元数之间的夹角
		const theta = theta0 * t; // 插值后的夹角
		const sinTheta0 = Math.sin(theta0);
		const sinTheta = Math.sin(theta);

		const s0 = Math.cos(theta) - (dot * sinTheta) / sinTheta0;
		const s1 = sinTheta / sinTheta0;

		return new Quaternion(
			s0 * q1.x + s1 * q2Adjusted.x,
			s0 * q1.y + s1 * q2Adjusted.y,
			s0 * q1.z + s1 * q2Adjusted.z,
			s0 * q1.w + s1 * q2Adjusted.w
		);
	}

	public rotatePoint(point: IVector3): IVector3 {
		const { x, y, z } = point;
		const q = this;

		// 转换为四元数形式
		const p = new Quaternion(x, y, z, 0);

		// 计算旋转: q * p * q^-1
		const qConjugate = new Quaternion(-q.x, -q.y, -q.z, q.w);
		const temp = Quaternion.multiply(q, p);
		const result = Quaternion.multiply(temp, qConjugate);

		return { x: result.x, y: result.y, z: result.z };
	}

	public normalize(): this {
		const length = Math.sqrt(
			this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w
		);
		if (length === 0) return this;

		this.x /= length;
		this.y /= length;
		this.z /= length;
		this.w /= length;
		return this;
	}
}
