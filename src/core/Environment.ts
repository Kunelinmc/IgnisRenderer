import { EventEmitter } from "./EventEmitter";
import type { Texture } from "./Texture";

export interface EnvironmentTintLinear {
	r: number;
	g: number;
	b: number;
}

export interface EnvironmentParams {
	backgroundEnabled?: boolean;
	lightingEnabled?: boolean;
	backgroundTexture?: Texture | null;
	iblTexture?: Texture | null;
	backgroundStrength?: number;
	diffuseStrength?: number;
	specularStrength?: number;
	backgroundTintLinear?: EnvironmentTintLinear;
	backgroundExposure?: number;
}

const DEFAULT_TINT_LINEAR: EnvironmentTintLinear = {
	r: 1,
	g: 1,
	b: 1,
};

export interface EnvironmentEvents {
	change: [];
	[key: string]: any[];
}

export class Environment extends EventEmitter<EnvironmentEvents> {
	private _backgroundEnabled: boolean;
	private _lightingEnabled: boolean;
	private _backgroundTexture: Texture | null;
	private _iblTexture: Texture | null;
	private _backgroundStrength: number;
	private _diffuseStrength: number;
	private _specularStrength: number;
	private _backgroundTintLinear: EnvironmentTintLinear;
	private _backgroundExposure: number;

	constructor(params: EnvironmentParams = {}) {
		super();
		this._backgroundEnabled = params.backgroundEnabled ?? true;
		this._lightingEnabled = params.lightingEnabled ?? true;
		this._backgroundTexture = params.backgroundTexture ?? null;
		this._iblTexture = params.iblTexture ?? null;
		this._backgroundStrength = clampNonNegativeNumber(
			params.backgroundStrength,
			1
		);
		this._diffuseStrength = clampNonNegativeNumber(params.diffuseStrength, 1);
		this._specularStrength = clampNonNegativeNumber(params.specularStrength, 1);
		this._backgroundTintLinear = sanitizeTintLinear(params.backgroundTintLinear);
		this._backgroundExposure = clampPositiveNumber(params.backgroundExposure, 1);
	}

	public get backgroundEnabled(): boolean {
		return this._backgroundEnabled;
	}

	public set backgroundEnabled(value: boolean) {
		const next = value !== false;
		if (this._backgroundEnabled === next) return;
		this._backgroundEnabled = next;
		this._notifyChanged();
	}

	public get lightingEnabled(): boolean {
		return this._lightingEnabled;
	}

	public set lightingEnabled(value: boolean) {
		const next = value !== false;
		if (this._lightingEnabled === next) return;
		this._lightingEnabled = next;
		this._notifyChanged();
	}

	public get backgroundTexture(): Texture | null {
		return this._backgroundTexture;
	}

	public set backgroundTexture(value: Texture | null) {
		const next = value ?? null;
		if (this._backgroundTexture === next) return;
		this._backgroundTexture = next;
		this._notifyChanged();
	}

	public get iblTexture(): Texture | null {
		return this._iblTexture;
	}

	public set iblTexture(value: Texture | null) {
		const next = value ?? null;
		if (this._iblTexture === next) return;
		this._iblTexture = next;
		this._notifyChanged();
	}

	public get backgroundStrength(): number {
		return this._backgroundStrength;
	}

	public set backgroundStrength(value: number) {
		const next = clampNonNegativeNumber(value, this._backgroundStrength);
		if (this._backgroundStrength === next) return;
		this._backgroundStrength = next;
		this._notifyChanged();
	}

	public get diffuseStrength(): number {
		return this._diffuseStrength;
	}

	public set diffuseStrength(value: number) {
		const next = clampNonNegativeNumber(value, this._diffuseStrength);
		if (this._diffuseStrength === next) return;
		this._diffuseStrength = next;
		this._notifyChanged();
	}

	public get specularStrength(): number {
		return this._specularStrength;
	}

	public set specularStrength(value: number) {
		const next = clampNonNegativeNumber(value, this._specularStrength);
		if (this._specularStrength === next) return;
		this._specularStrength = next;
		this._notifyChanged();
	}

	public get backgroundTintLinear(): EnvironmentTintLinear {
		return {
			r: this._backgroundTintLinear.r,
			g: this._backgroundTintLinear.g,
			b: this._backgroundTintLinear.b,
		};
	}

	public set backgroundTintLinear(value: EnvironmentTintLinear) {
		const next = sanitizeTintLinear(value);
		if (
			this._backgroundTintLinear.r === next.r &&
			this._backgroundTintLinear.g === next.g &&
			this._backgroundTintLinear.b === next.b
		) {
			return;
		}
		this._backgroundTintLinear = next;
		this._notifyChanged();
	}

	public get backgroundExposure(): number {
		return this._backgroundExposure;
	}

	public set backgroundExposure(value: number) {
		const next = clampPositiveNumber(value, this._backgroundExposure);
		if (this._backgroundExposure === next) return;
		this._backgroundExposure = next;
		this._notifyChanged();
	}

	private _notifyChanged(): void {
		this.emit("change");
	}
}

function clampNonNegativeNumber(value: number, fallback: number): number {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(0, value);
}

function clampPositiveNumber(value: number, fallback: number): number {
	if (!Number.isFinite(value)) {
		return fallback;
	}
	return Math.max(1e-6, value);
}

function sanitizeTintLinear(
	value: EnvironmentTintLinear | undefined
): EnvironmentTintLinear {
	return {
		r: clamp01(value?.r ?? DEFAULT_TINT_LINEAR.r),
		g: clamp01(value?.g ?? DEFAULT_TINT_LINEAR.g),
		b: clamp01(value?.b ?? DEFAULT_TINT_LINEAR.b),
	};
}

function clamp01(value: number): number {
	if (!Number.isFinite(value)) {
		return 1;
	}
	return Math.min(1, Math.max(0, value));
}
