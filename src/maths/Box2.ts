import type { IVector2 } from "./types";
import { Vector2 } from "./Vector2";

export class Box2 {
	constructor(
		public min: Vector2 = new Vector2(Infinity, Infinity),
		public max: Vector2 = new Vector2(-Infinity, -Infinity)
	) {}

	public set(min: IVector2, max: IVector2): this {
		this.min.copy(min);
		this.max.copy(max);
		return this;
	}

	public expandByPoint(point: IVector2): this {
		this.min.x = Math.min(this.min.x, point.x);
		this.min.y = Math.min(this.min.y, point.y);
		this.max.x = Math.max(this.max.x, point.x);
		this.max.y = Math.max(this.max.y, point.y);
		return this;
	}

	public containsPoint(point: IVector2): boolean {
		return (
			point.x >= this.min.x &&
			point.x <= this.max.x &&
			point.y >= this.min.y &&
			point.y <= this.max.y
		);
	}

	public intersectsBox(box: Box2): boolean {
		return (
			box.max.x >= this.min.x &&
			box.min.x <= this.max.x &&
			box.max.y >= this.min.y &&
			box.min.y <= this.max.y
		);
	}

	public clone(): Box2 {
		return new Box2(this.min.clone(), this.max.clone());
	}
}
