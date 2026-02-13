/**
 * Vector2 class and utility functions
 */

import type { IVector2 } from "./types";

export class Vector2 implements IVector2 {
	constructor(
		public x: number = 0,
		public y: number = 0
	) {}

	public set(x: number, y: number): this {
		this.x = x;
		this.y = y;
		return this;
	}

	public copy(v: IVector2): this {
		this.x = v.x;
		this.y = v.y;
		return this;
	}

	public clone(): Vector2 {
		return new Vector2(this.x, this.y);
	}

	public add(v: IVector2): this {
		this.x += v.x;
		this.y += v.y;
		return this;
	}

	public sub(v: IVector2): this {
		this.x -= v.x;
		this.y -= v.y;
		return this;
	}

	public scale(s: number): this {
		this.x *= s;
		this.y *= s;
		return this;
	}

	public dot(v: IVector2): number {
		return this.x * v.x + this.y * v.y;
	}

	public length(): number {
		return Math.hypot(this.x, this.y);
	}

	public normalize(): this {
		const len = this.length() || 1;
		return this.scale(1 / len);
	}

	// Static methods for functional style
	public static normalize(v: IVector2): Vector2 {
		return new Vector2(v.x, v.y).normalize();
	}

	public static dot(a: IVector2, b: IVector2): number {
		return a.x * b.x + a.y * b.y;
	}

	public static add(a: IVector2, b: IVector2): Vector2 {
		return new Vector2(a.x + b.x, a.y + b.y);
	}

	public static sub(a: IVector2, b: IVector2): Vector2 {
		return new Vector2(a.x - b.x, a.y - b.y);
	}

	public static scale(v: IVector2, s: number): Vector2 {
		return new Vector2(v.x * s, v.y * s);
	}

	public static length(v: IVector2): number {
		return Math.hypot(v.x, v.y);
	}
}
