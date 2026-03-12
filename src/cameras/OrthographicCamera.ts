import { Camera, CameraType } from "./Camera";
import { Matrix4 } from "../maths/Matrix4";

export interface OrthographicBounds {
	left: number;
	right: number;
	bottom: number;
	top: number;
}

/**
 * An Orthographic Camera using a size-based orthographic projection.
 */
export class OrthographicCamera extends Camera {
	/** Provide the vertical size of the visible area. The horizontal size is computed via the aspect ratio. */
	public size: number = 100;
	/** Optional override for the left clipping plane. If null, size/aspect-derived value is used. */
	public left: number | null = null;
	/** Optional override for the right clipping plane. If null, size/aspect-derived value is used. */
	public right: number | null = null;
	/** Optional override for the bottom clipping plane. If null, size-derived value is used. */
	public bottom: number | null = null;
	/** Optional override for the top clipping plane. If null, size-derived value is used. */
	public top: number | null = null;

	constructor(size: number = 100) {
		super();
		this.type = CameraType.Orthographic;
		this.size = size;

		this.updateMatrices();
	}

	public setBounds(
		left: number,
		right: number,
		bottom: number,
		top: number
	): this {
		this.left = left;
		this.right = right;
		this.bottom = bottom;
		this.top = top;
		return this;
	}

	public clearBounds(): this {
		this.left = null;
		this.right = null;
		this.bottom = null;
		this.top = null;
		return this;
	}

	public getBounds(): OrthographicBounds {
		const size = Number.isFinite(this.size) ? this.size : 100;
		const aspectRatio = Number.isFinite(this.aspectRatio)
			? this.aspectRatio
			: 16 / 9;
		const halfHeight = size / 2;
		const halfWidth = halfHeight * aspectRatio;

		return {
			left: typeof this.left === "number" ? this.left : -halfWidth,
			right: typeof this.right === "number" ? this.right : halfWidth,
			bottom: typeof this.bottom === "number" ? this.bottom : -halfHeight,
			top: typeof this.top === "number" ? this.top : halfHeight,
		};
	}

	public override calculateProjectionMatrix(): Matrix4 {
		const bounds = this.getBounds();

		if (!Number.isFinite(bounds.left) || !Number.isFinite(bounds.right)) {
			throw new Error("OrthographicCamera bounds left/right must be finite numbers");
		}
		if (!Number.isFinite(bounds.bottom) || !Number.isFinite(bounds.top)) {
			throw new Error("OrthographicCamera bounds bottom/top must be finite numbers");
		}
		if (Math.abs(bounds.right - bounds.left) <= 1e-8) {
			throw new Error("OrthographicCamera requires right != left");
		}
		if (Math.abs(bounds.top - bounds.bottom) <= 1e-8) {
			throw new Error("OrthographicCamera requires top != bottom");
		}

		return Matrix4.ortho(
			bounds.left,
			bounds.right,
			bounds.bottom,
			bounds.top,
			this.near,
			this.far
		);
	}
}
