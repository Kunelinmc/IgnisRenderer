import type { Node } from "../core/Node";
import { Matrix4 } from "../maths/Matrix4";

export interface SkeletonOptions {
	name?: string;
	joints: Node[];
	inverseBindMatrices: Matrix4[];
}

export class Skeleton {
	public readonly name: string;
	public readonly joints: Node[];
	public readonly inverseBindMatrices: Matrix4[];
	public readonly jointMatrices: Matrix4[];

	constructor(options: SkeletonOptions) {
		if (options.joints.length !== options.inverseBindMatrices.length) {
			throw new Error(
				`Skeleton "${options.name ?? "skeleton"}" joints/inverseBindMatrices length mismatch`
			);
		}
		this.name = options.name ?? "skeleton";
		this.joints = [...options.joints];
		this.inverseBindMatrices = options.inverseBindMatrices.map((matrix) =>
			matrix.clone()
		);
		this.jointMatrices = this.joints.map(() => Matrix4.identity());
	}

	public get jointCount(): number {
		return this.joints.length;
	}

	public updateJointMatrices(): void {
		for (let i = 0; i < this.joints.length; i++) {
			Matrix4.multiply(
				this.joints[i].worldMatrix,
				this.inverseBindMatrices[i],
				this.jointMatrices[i]
			);
		}
	}

	public toFloat32Array(out?: Float32Array): Float32Array {
		const target = out ?? new Float32Array(this.jointCount * 16);
		for (let i = 0; i < this.jointMatrices.length; i++) {
			const matrix = this.jointMatrices[i].elements;
			const offset = i * 16;
			target[offset] = matrix[0][0];
			target[offset + 1] = matrix[1][0];
			target[offset + 2] = matrix[2][0];
			target[offset + 3] = matrix[3][0];
			target[offset + 4] = matrix[0][1];
			target[offset + 5] = matrix[1][1];
			target[offset + 6] = matrix[2][1];
			target[offset + 7] = matrix[3][1];
			target[offset + 8] = matrix[0][2];
			target[offset + 9] = matrix[1][2];
			target[offset + 10] = matrix[2][2];
			target[offset + 11] = matrix[3][2];
			target[offset + 12] = matrix[0][3];
			target[offset + 13] = matrix[1][3];
			target[offset + 14] = matrix[2][3];
			target[offset + 15] = matrix[3][3];
		}
		return target;
	}
}
