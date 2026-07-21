import type { Renderer } from "../rendering/Renderer";
import { Camera } from "../cameras/Camera";
import { OrbitCamera } from "../cameras/OrbitCamera";
import { perlinNoise1D } from "../maths/Noise";
import { Quaternion } from "../maths/Quaternion";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";

const TAU = Math.PI * 2;
const EPSILON = 1e-8;
// 1D Perlin peaks near +/-0.5, so preserve the configured shake amplitude scale.
const PERLIN_AMPLITUDE_SCALE = 2;
const POSITION_X_NOISE_SEED = 101;
const POSITION_Y_NOISE_SEED = 211;
const POSITION_Z_NOISE_SEED = 307;
const ROTATION_X_NOISE_SEED = 401;
const ROTATION_Y_NOISE_SEED = 503;
const ROTATION_Z_NOISE_SEED = 601;

const DEFAULT_INTENSITY = 0.8;
const DEFAULT_DURATION_SECONDS = 0.35;
const DEFAULT_FREQUENCY_HZ = 22;
const DEFAULT_FALLOFF_EXPONENT = 2.2;

const DEFAULT_POSITION_AMPLITUDE: IVector3 = {
	x: 0.18,
	y: 0.12,
	z: 0.1,
};

const DEFAULT_ROTATION_AMPLITUDE: IVector3 = {
	x: 0.014,
	y: 0.016,
	z: 0.01,
};

/**
 * Default configuration for CameraShakePlugin.
 */
export interface CameraShakePluginOptions {
	defaultIntensity?: number;
	defaultDurationSeconds?: number;
	defaultFrequencyHz?: number;
	defaultFalloffExponent?: number;
	defaultPositionAmplitude?: IVector3;
	defaultRotationAmplitude?: IVector3;
}

/**
 * Runtime shake impulse payload.
 */
export interface CameraShakeImpulse {
	intensity?: number;
	durationSeconds?: number;
	frequencyHz?: number;
	positionAmplitude?: IVector3;
	rotationAmplitude?: IVector3;
}

interface ActiveCameraShake {
	elapsedSeconds: number;
	intensity: number;
	durationSeconds: number;
	frequencyHz: number;
	falloffExponent: number;
	positionAmplitude: Vector3;
	rotationAmplitude: Vector3;
}

/**
 * Event-driven camera shake plugin.
 *
 * Integrates with Renderer "tick/postanimation/frameend" hooks so shake is:
 * - updated in simulation time,
 * - applied before transform/update and scene build,
 * - restored after frame rendering to prevent camera drift.
 */
export class CameraShakePlugin {
	private _renderer: Renderer | null = null;
	private _cameraOverride: Camera | null = null;

	private _activeShakes: ActiveCameraShake[] = [];

	private _defaultIntensity = DEFAULT_INTENSITY;
	private _defaultDurationSeconds = DEFAULT_DURATION_SECONDS;
	private _defaultFrequencyHz = DEFAULT_FREQUENCY_HZ;
	private _defaultFalloffExponent = DEFAULT_FALLOFF_EXPONENT;
	private _defaultPositionAmplitude = new Vector3(
		DEFAULT_POSITION_AMPLITUDE.x,
		DEFAULT_POSITION_AMPLITUDE.y,
		DEFAULT_POSITION_AMPLITUDE.z
	);
	private _defaultRotationAmplitude = new Vector3(
		DEFAULT_ROTATION_AMPLITUDE.x,
		DEFAULT_ROTATION_AMPLITUDE.y,
		DEFAULT_ROTATION_AMPLITUDE.z
	);

	private _applied = false;
	private _appliedCamera: Camera | null = null;
	private _basePosition = new Vector3();
	private _baseQuaternion = new Quaternion();
	private _baseTarget = new Vector3();
	private _baseUp = new Vector3(0, 1, 0);
	private _shakeQuaternion = new Quaternion();
	private _orbitPitchQuaternion = new Quaternion();
	private _orbitYawQuaternion = new Quaternion();
	private _orbitRollQuaternion = new Quaternion();
	private _orbitRotationQuaternion = new Quaternion();
	private _orbitRotatedOffset = new Vector3();
	private _orbitRotatedUp = new Vector3();

	private _onTick = (event: { now: number; deltaTime: number }): void => {
		if (this._activeShakes.length === 0) return;

		const deltaTimeSeconds = Math.max(0, event.deltaTime) / 1000;
		for (let i = this._activeShakes.length - 1; i >= 0; i--) {
			const shake = this._activeShakes[i];
			shake.elapsedSeconds += deltaTimeSeconds;
			if (shake.elapsedSeconds >= shake.durationSeconds) {
				this._activeShakes.splice(i, 1);
			}
		}

		if (this._activeShakes.length > 0) {
			this._renderer?.requestRender("camera");
		}
	};

	private _onPostAnimation = (): void => {
		this._applyShakeIfNeeded();
	};

	private _onFrameEnd = (): void => {
		this._restoreApplied();
	};

	constructor(options: CameraShakePluginOptions = {}) {
		this._applyDefaults(options);
	}

	/**
	 * Whether plugin is currently attached to a renderer.
	 */
	public get isAttached(): boolean {
		return this._renderer !== null;
	}

	/**
	 * Whether shake is currently active for upcoming frames.
	 */
	public get isActive(): boolean {
		return this._activeShakes.length > 0 || this._applied;
	}

	/**
	 * Attach plugin to renderer lifecycle events.
	 */
	public attach(renderer: Renderer, camera: Camera | null = null): this {
		if (this._renderer === renderer && this._cameraOverride === camera) {
			return this;
		}
		this.detach();

		this._renderer = renderer;
		this._cameraOverride = camera;
		renderer.on("tick", this._onTick);
		renderer.on("postanimation", this._onPostAnimation);
		renderer.on("frameend", this._onFrameEnd);
		return this;
	}

	/**
	 * Detach plugin and restore pending camera override immediately.
	 */
	public detach(): void {
		if (this._renderer) {
			this._renderer.off("tick", this._onTick);
			this._renderer.off("postanimation", this._onPostAnimation);
			this._renderer.off("frameend", this._onFrameEnd);
		}
		this._restoreApplied();
		this._renderer = null;
		this._cameraOverride = null;
		this._activeShakes.length = 0;
	}

	/**
	 * Alias of detach for explicit lifecycle cleanup.
	 */
	public destroy(): void {
		this.detach();
	}

	/**
	 * Override target camera; null means "use renderer.camera".
	 */
	public setCamera(camera: Camera | null): this {
		this._cameraOverride = camera;
		return this;
	}

	/**
	 * Start or stack an independently timed camera shake impulse.
	 */
	public trigger(impulse: CameraShakeImpulse = {}): void {
		const intensity = clamp01(
			Number.isFinite(impulse.intensity) ?
				Number(impulse.intensity)
			:	this._defaultIntensity
		);
		if (intensity <= 0) return;

		const durationSeconds = sanitizePositive(
			impulse.durationSeconds,
			this._defaultDurationSeconds
		);
		const frequencyHz = sanitizePositive(
			impulse.frequencyHz,
			this._defaultFrequencyHz
		);

		const positionAmplitude = hasFiniteVector(impulse.positionAmplitude) ?
				impulse.positionAmplitude
			:	{
					x: this._defaultPositionAmplitude.x,
					y: this._defaultPositionAmplitude.y,
					z: this._defaultPositionAmplitude.z,
				};
		const rotationAmplitude = hasFiniteVector(impulse.rotationAmplitude) ?
				impulse.rotationAmplitude
			:	{
					x: this._defaultRotationAmplitude.x,
					y: this._defaultRotationAmplitude.y,
					z: this._defaultRotationAmplitude.z,
				};

		this._activeShakes.push({
			elapsedSeconds: 0,
			intensity,
			durationSeconds,
			frequencyHz,
			falloffExponent: this._defaultFalloffExponent,
			positionAmplitude: new Vector3(
				positionAmplitude.x,
				positionAmplitude.y,
				positionAmplitude.z
			),
			rotationAmplitude: new Vector3(
				rotationAmplitude.x,
				rotationAmplitude.y,
				rotationAmplitude.z
			),
		});
		this._renderer?.requestRender("camera");
	}

	/**
	 * Stop ongoing shake and restore camera immediately if needed.
	 */
	public stop(): void {
		this._activeShakes.length = 0;
		this._restoreApplied();
	}

	private _applyDefaults(options: CameraShakePluginOptions): void {
		this._defaultIntensity = clamp01(
			Number.isFinite(options.defaultIntensity) ?
				Number(options.defaultIntensity)
			:	DEFAULT_INTENSITY
		);
		this._defaultDurationSeconds = sanitizePositive(
			options.defaultDurationSeconds,
			DEFAULT_DURATION_SECONDS
		);
		this._defaultFrequencyHz = sanitizePositive(
			options.defaultFrequencyHz,
			DEFAULT_FREQUENCY_HZ
		);
		this._defaultFalloffExponent = sanitizePositive(
			options.defaultFalloffExponent,
			DEFAULT_FALLOFF_EXPONENT
		);
		if (hasFiniteVector(options.defaultPositionAmplitude)) {
			this._defaultPositionAmplitude.copy(options.defaultPositionAmplitude);
		}
		if (hasFiniteVector(options.defaultRotationAmplitude)) {
			this._defaultRotationAmplitude.copy(options.defaultRotationAmplitude);
		}
	}

	private _resolveCamera(): Camera | null {
		if (this._cameraOverride) return this._cameraOverride;
		return this._renderer?.camera ?? null;
	}

	private _applyShakeIfNeeded(): void {
		if (this._applied) {
			this._restoreApplied();
		}
		if (this._activeShakes.length === 0) return;

		const camera = this._resolveCamera();
		if (!camera) return;

		let positionX = 0;
		let positionY = 0;
		let positionZ = 0;
		let rotationX = 0;
		let rotationY = 0;
		let rotationZ = 0;
		let hasContribution = false;

		for (const shake of this._activeShakes) {
			const duration = Math.max(shake.durationSeconds, 1e-4);
			const t = Math.min(1, shake.elapsedSeconds / duration);
			const envelope = Math.pow(1 - t, shake.falloffExponent);
			const gain = shake.intensity * envelope;
			if (gain <= 1e-6) continue;

			const noiseCoordinate =
				shake.elapsedSeconds * shake.frequencyHz * TAU;
			// Independent seeds prevent position and rotation from sampling shifted
			// segments of the same Perlin field.
			positionX +=
				shake.positionAmplitude.x * gain *
				perlinNoise1D(
					noiseCoordinate + 0.713,
					POSITION_X_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			positionY +=
				shake.positionAmplitude.y * gain *
				perlinNoise1D(
					noiseCoordinate + 2.173,
					POSITION_Y_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			positionZ +=
				shake.positionAmplitude.z * gain *
				perlinNoise1D(
					noiseCoordinate + 4.631,
					POSITION_Z_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			rotationX +=
				shake.rotationAmplitude.x * gain *
				perlinNoise1D(
					noiseCoordinate + 1.371,
					ROTATION_X_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			rotationY +=
				shake.rotationAmplitude.y * gain *
				perlinNoise1D(
					noiseCoordinate + 3.019,
					ROTATION_Y_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			rotationZ +=
				shake.rotationAmplitude.z * gain *
				perlinNoise1D(
					noiseCoordinate + 5.707,
					ROTATION_Z_NOISE_SEED
				) * PERLIN_AMPLITUDE_SCALE;
			hasContribution = true;
		}

		if (!hasContribution) return;

		this._basePosition.copy(camera.position);
		this._appliedCamera = camera;

		if (camera instanceof OrbitCamera) {
			this._baseTarget.copy(camera.target);
			this._baseUp.copy(camera.up);
			this._applyOrbitShake(
				camera,
				positionX,
				positionY,
				positionZ,
				rotationX,
				rotationY,
				rotationZ
			);
		} else {
			this._baseQuaternion.copy(camera.quaternion);
			camera.position.x += positionX;
			camera.position.y += positionY;
			camera.position.z += positionZ;
			this._shakeQuaternion.fromEuler(rotationX, rotationY, rotationZ).normalize();
			multiplyQuaternions(
				this._shakeQuaternion,
				this._baseQuaternion,
				camera.quaternion
			);
		}

		this._applied = true;
	}

	private _applyOrbitShake(
		camera: OrbitCamera,
		positionX: number,
		positionY: number,
		positionZ: number,
		rotationX: number,
		rotationY: number,
		rotationZ: number
	): void {
		let forwardX = camera.target.x - camera.position.x;
		let forwardY = camera.target.y - camera.position.y;
		let forwardZ = camera.target.z - camera.position.z;
		let forwardLength = Math.hypot(forwardX, forwardY, forwardZ);
		if (forwardLength <= EPSILON) {
			forwardX = 0;
			forwardY = 0;
			forwardZ = -1;
			forwardLength = 1;
		}
		forwardX /= forwardLength;
		forwardY /= forwardLength;
		forwardZ /= forwardLength;

		let rightX = forwardY * camera.up.z - forwardZ * camera.up.y;
		let rightY = forwardZ * camera.up.x - forwardX * camera.up.z;
		let rightZ = forwardX * camera.up.y - forwardY * camera.up.x;
		let rightLength = Math.hypot(rightX, rightY, rightZ);
		if (rightLength <= EPSILON) {
			rightX = 1;
			rightY = 0;
			rightZ = 0;
			rightLength = 1;
		}
		rightX /= rightLength;
		rightY /= rightLength;
		rightZ /= rightLength;

		let upX = rightY * forwardZ - rightZ * forwardY;
		let upY = rightZ * forwardX - rightX * forwardZ;
		let upZ = rightX * forwardY - rightY * forwardX;
		const upLength = Math.hypot(upX, upY, upZ);
		if (upLength > EPSILON) {
			upX /= upLength;
			upY /= upLength;
			upZ /= upLength;
		}

		const translateX =
			rightX * positionX + upX * positionY + forwardX * positionZ;
		const translateY =
			rightY * positionX + upY * positionY + forwardY * positionZ;
		const translateZ =
			rightZ * positionX + upZ * positionY + forwardZ * positionZ;

		const pivotX = this._baseTarget.x + translateX;
		const pivotY = this._baseTarget.y + translateY;
		const pivotZ = this._baseTarget.z + translateZ;

		setQuaternionFromAxisAngle(
			this._orbitPitchQuaternion,
			rightX,
			rightY,
			rightZ,
			rotationX
		);
		setQuaternionFromAxisAngle(
			this._orbitYawQuaternion,
			upX,
			upY,
			upZ,
			rotationY
		);
		setQuaternionFromAxisAngle(
			this._orbitRollQuaternion,
			forwardX,
			forwardY,
			forwardZ,
			rotationZ
		);
		multiplyQuaternions(
			this._orbitYawQuaternion,
			this._orbitPitchQuaternion,
			this._orbitRotationQuaternion
		);
		multiplyQuaternions(
			this._orbitRollQuaternion,
			this._orbitRotationQuaternion,
			this._orbitRotationQuaternion
		);

		rotateVectorByQuaternion(
			this._basePosition.x - this._baseTarget.x,
			this._basePosition.y - this._baseTarget.y,
			this._basePosition.z - this._baseTarget.z,
			this._orbitRotationQuaternion,
			this._orbitRotatedOffset
		);
		camera.position.x = pivotX + this._orbitRotatedOffset.x;
		camera.position.y = pivotY + this._orbitRotatedOffset.y;
		camera.position.z = pivotZ + this._orbitRotatedOffset.z;
		camera.target.x = pivotX;
		camera.target.y = pivotY;
		camera.target.z = pivotZ;

		rotateVectorByQuaternion(
			this._baseUp.x,
			this._baseUp.y,
			this._baseUp.z,
			this._orbitRotationQuaternion,
			this._orbitRotatedUp
		);
		const rotatedUpLength = Math.hypot(
			this._orbitRotatedUp.x,
			this._orbitRotatedUp.y,
			this._orbitRotatedUp.z
		);
		if (rotatedUpLength > EPSILON) {
			const invUpLength = 1 / rotatedUpLength;
			camera.up.x = this._orbitRotatedUp.x * invUpLength;
			camera.up.y = this._orbitRotatedUp.y * invUpLength;
			camera.up.z = this._orbitRotatedUp.z * invUpLength;
		} else {
			camera.up.copy(this._baseUp);
		}
	}

	private _restoreApplied(): void {
		if (!this._applied || !this._appliedCamera) return;

		const camera = this._appliedCamera;
		camera.position.copy(this._basePosition);

		if (camera instanceof OrbitCamera) {
			camera.target.copy(this._baseTarget);
			camera.up.copy(this._baseUp);
		} else {
			camera.quaternion.copy(this._baseQuaternion);
		}
		camera.updateMatrices();

		this._applied = false;
		this._appliedCamera = null;
	}
}

function sanitizePositive(
	value: number | undefined,
	fallback: number
): number {
	if (!Number.isFinite(value)) return fallback;
	return Math.max(1e-4, Number(value));
}

function clamp01(value: number): number {
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}

function hasFiniteVector(value: IVector3 | undefined): value is IVector3 {
	if (!value) return false;
	return (
		Number.isFinite(value.x) &&
		Number.isFinite(value.y) &&
		Number.isFinite(value.z)
	);
}

function multiplyQuaternions(
	left: Quaternion,
	right: Quaternion,
	out: Quaternion
): void {
	const x =
		left.w * right.x +
		left.x * right.w +
		left.y * right.z -
		left.z * right.y;
	const y =
		left.w * right.y -
		left.x * right.z +
		left.y * right.w +
		left.z * right.x;
	const z =
		left.w * right.z +
		left.x * right.y -
		left.y * right.x +
		left.z * right.w;
	const w =
		left.w * right.w -
		left.x * right.x -
		left.y * right.y -
		left.z * right.z;
	out.set(x, y, z, w).normalize();
}

function setQuaternionFromAxisAngle(
	out: Quaternion,
	axisX: number,
	axisY: number,
	axisZ: number,
	angle: number
): void {
	const axisLength = Math.hypot(axisX, axisY, axisZ);
	if (axisLength <= EPSILON || Math.abs(angle) <= EPSILON) {
		out.set(0, 0, 0, 1);
		return;
	}
	const invAxisLength = 1 / axisLength;
	const halfAngle = angle * 0.5;
	const sinHalf = Math.sin(halfAngle);
	const cosHalf = Math.cos(halfAngle);
	out.set(
		axisX * invAxisLength * sinHalf,
		axisY * invAxisLength * sinHalf,
		axisZ * invAxisLength * sinHalf,
		cosHalf
	).normalize();
}

function rotateVectorByQuaternion(
	x: number,
	y: number,
	z: number,
	rotation: Quaternion,
	out: IVector3
): void {
	const qx = rotation.x;
	const qy = rotation.y;
	const qz = rotation.z;
	const qw = rotation.w;

	const ix = qw * x + qy * z - qz * y;
	const iy = qw * y + qz * x - qx * z;
	const iz = qw * z + qx * y - qy * x;
	const iw = -qx * x - qy * y - qz * z;

	out.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
	out.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
	out.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;
}
