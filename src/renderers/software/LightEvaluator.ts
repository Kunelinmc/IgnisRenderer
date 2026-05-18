import { linearToSRGB } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import { SH } from "../../maths/SH";
import { Vector3 } from "../../maths/Vector3";
import type { IVector3 } from "../../maths/types";
import type { RGB } from "../../foundation/Color";
import { LightType } from "../../lights/Light";
import type { AmbientLight } from "../../lights/AmbientLight";
import type { AreaLight } from "../../lights/AreaLight";
import type { DirectionalLight } from "../../lights/DirectionalLight";
import type { PointLight } from "../../lights/PointLight";
import type { LightProbe } from "../../lights/LightProbe";
import type { ReflectionProbe } from "../../lights/ReflectionProbe";
import type { SpotLight } from "../../lights/SpotLight";

type SceneLight =
	| AmbientLight
	| DirectionalLight
	| PointLight
	| SpotLight
	| LightProbe
	| ReflectionProbe
	| AreaLight;

export interface SurfacePoint {
	position: IVector3;
	normal?: IVector3;
}

export interface LightContribution {
	type: "ambient" | "direct" | "irradiance";
	color: RGB;
	intensity?: number;
	direction?: IVector3;
}

export interface MutableLightContribution extends LightContribution {
	color: RGB;
	direction: IVector3;
}

const ORIGIN = { x: 0, y: 0, z: 0 };
const DEFAULT_DIRECTION = { x: 0, y: 1, z: 0 };
const LIGHT_PROBE_DC_IRRADIANCE_SCALE = Math.PI * 0.282095;
const AREA_LIGHT_SAMPLE_GRID_SIZE = 3;
const AREA_LIGHT_SAMPLE_COUNT =
	AREA_LIGHT_SAMPLE_GRID_SIZE * AREA_LIGHT_SAMPLE_GRID_SIZE;

export function createLightContribution(): MutableLightContribution {
	return {
		type: "ambient",
		color: { r: 255, g: 255, b: 255 },
		intensity: 1,
		direction: {
			x: DEFAULT_DIRECTION.x,
			y: DEFAULT_DIRECTION.y,
			z: DEFAULT_DIRECTION.z,
		},
	};
}

export function evaluateLightContribution(
	light: SceneLight,
	surface: SurfacePoint,
	out?: MutableLightContribution
): LightContribution | null {
	const contribution = out ?? createLightContribution();

	switch (light.type) {
		case LightType.Ambient:
			return evaluateAmbientLight(light, contribution);
		case LightType.Directional:
			return evaluateDirectionalLight(light, contribution);
		case LightType.Point:
			return evaluatePointLight(light, surface, contribution);
		case LightType.Spot:
			return evaluateSpotLight(light, surface, contribution);
		case LightType.LightProbe:
			return evaluateLightProbe(light, surface, contribution);
		case LightType.ReflectionProbe:
			return null;
		case LightType.RectArea:
			return evaluateAreaLight(light, surface, contribution);
		default:
			return null;
	}
}

function evaluateAmbientLight(
	light: AmbientLight,
	out: MutableLightContribution
): LightContribution {
	return writeContribution(out, {
		type: "ambient",
		color: light.color,
		intensity: light.intensity,
	});
}

function evaluateDirectionalLight(
	light: DirectionalLight,
	out: MutableLightContribution
): LightContribution {
	const direction = light.getWorldLightDirection(out.direction);
	direction.x = -direction.x;
	direction.y = -direction.y;
	direction.z = -direction.z;

	return writeContribution(out, {
		type: "direct",
		color: light.color,
		intensity: light.intensity,
		direction,
	});
}

function evaluatePointLight(
	light: PointLight,
	surface: SurfacePoint,
	out: MutableLightContribution
): LightContribution | null {
	const position = requireSurfacePosition(surface);
	const lightPos = light.getWorldLightPosition();
	const dx = lightPos.x - position.x;
	const dy = lightPos.y - position.y;
	const dz = lightPos.z - position.z;
	const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

	if (distance > light.range) return null;

	const distanceSq = distance * distance;
	const rangeSq = light.range * light.range;
	const rangeFactor = distanceSq / rangeSq;
	const smoothFactor = Math.max(0, 1 - rangeFactor * rangeFactor);
	const attenuation = (smoothFactor * smoothFactor) / (distanceSq + 1.0);
	const direction = out.direction;

	if (distance > 0) {
		direction.x = dx / distance;
		direction.y = dy / distance;
		direction.z = dz / distance;
	} else {
		direction.x = DEFAULT_DIRECTION.x;
		direction.y = DEFAULT_DIRECTION.y;
		direction.z = DEFAULT_DIRECTION.z;
	}

	return writeContribution(out, {
		type: "direct",
		color: light.color,
		intensity: light.intensity * attenuation,
		direction,
	});
}

function evaluateSpotLight(
	light: SpotLight,
	surface: SurfacePoint,
	out: MutableLightContribution
): LightContribution | null {
	const position = requireSurfacePosition(surface);
	const lightPos = light.getWorldLightPosition();
	const lightDir = light.getWorldLightDirection();
	const dx = lightPos.x - position.x;
	const dy = lightPos.y - position.y;
	const dz = lightPos.z - position.z;
	const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

	if (distance > light.range) return null;

	const direction = out.direction;
	if (distance > 0) {
		direction.x = dx / distance;
		direction.y = dy / distance;
		direction.z = dz / distance;
	} else {
		direction.x = DEFAULT_DIRECTION.x;
		direction.y = DEFAULT_DIRECTION.y;
		direction.z = DEFAULT_DIRECTION.z;
	}

	const lightToPointX = -direction.x;
	const lightToPointY = -direction.y;
	const lightToPointZ = -direction.z;
	const cosTheta =
		lightToPointX * lightDir.x +
		lightToPointY * lightDir.y +
		lightToPointZ * lightDir.z;

	const outerCutoff = Math.cos(light.outerAngle);
	const innerCutoff = Math.cos(light.getInnerAngle());
	const cutoffRange = innerCutoff - outerCutoff;

	if (cosTheta < outerCutoff) return null;

	const spotIntensity = Math.max(
		0,
		Math.min(1, (cosTheta - outerCutoff) / (cutoffRange || 1e-6))
	);
	const distanceSq = distance * distance;
	const rangeSq = light.range * light.range;
	const rangeFactor = distanceSq / rangeSq;
	const smoothFactor = Math.max(0, 1 - rangeFactor * rangeFactor);
	const attenuation = (smoothFactor * smoothFactor) / (distanceSq + 1.0);

	return writeContribution(out, {
		type: "direct",
		color: light.color,
		intensity: light.intensity * attenuation * spotIntensity,
		direction,
	});
}

function evaluateLightProbe(
	light: LightProbe,
	surface: SurfacePoint,
	out: MutableLightContribution
): LightContribution | null {
	let irrR = 0;
	let irrG = 0;
	let irrB = 0;

	if (surface.normal) {
		const irr = SH.calculateIrradiance(surface.normal, light.sh);
		irrR = Math.max(0, irr.r);
		irrG = Math.max(0, irr.g);
		irrB = Math.max(0, irr.b);
	} else {
		const dc = light.sh[0];
		irrR = Math.max(0, dc.r * LIGHT_PROBE_DC_IRRADIANCE_SCALE);
		irrG = Math.max(0, dc.g * LIGHT_PROBE_DC_IRRADIANCE_SCALE);
		irrB = Math.max(0, dc.b * LIGHT_PROBE_DC_IRRADIANCE_SCALE);
	}

	if (irrR <= 0 && irrG <= 0 && irrB <= 0) return null;

	return writeContribution(out, {
		type: "irradiance",
		color: {
			r: toSrgb255(irrR),
			g: toSrgb255(irrG),
			b: toSrgb255(irrB),
		},
		intensity: 1,
	});
}

function evaluateAreaLight(
	light: AreaLight,
	surface: SurfacePoint,
	out: MutableLightContribution
): LightContribution | null {
	const surfacePos = requireSurfacePosition(surface);
	const width = Math.max(light.width, 0);
	const height = Math.max(light.height, 0);
	const range = Math.max(light.range, 0);
	if (width <= 0 || height <= 0 || range <= 0) return null;

	const worldMatrix = light.worldMatrix;
	const center = Matrix4.transformPoint(worldMatrix, { x: 0, y: 0, z: 0 });
	const right = Vector3.normalize(
		Matrix4.transformDirection(worldMatrix, { x: 1, y: 0, z: 0 })
	);
	const up = Vector3.normalize(
		Matrix4.transformDirection(worldMatrix, { x: 0, y: 0, z: 1 })
	);
	const normal = Vector3.normalize(
		Matrix4.transformDirection(worldMatrix, { x: 0, y: 1, z: 0 })
	);

	const relPos = Vector3.sub(surfacePos, center);
	const distToPlane = Vector3.dot(relPos, normal);
	if (distToPlane <= 0) return null;

	const direction = out.direction;
	let accumulatedX = 0;
	let accumulatedY = 0;
	let accumulatedZ = 0;
	let attenuation = 0;
	const cellHalfWidth = (width / AREA_LIGHT_SAMPLE_GRID_SIZE) * 0.5;
	const cellHalfHeight = (height / AREA_LIGHT_SAMPLE_GRID_SIZE) * 0.5;

	// Integrate finite emitter energy per cell, then expose one weighted
	// direction because the software light contribution API is directional.
	for (
		let sampleIndex = 0;
		sampleIndex < AREA_LIGHT_SAMPLE_COUNT;
		sampleIndex++
	) {
		const sampleX = sampleIndex % AREA_LIGHT_SAMPLE_GRID_SIZE;
		const sampleY = Math.floor(sampleIndex / AREA_LIGHT_SAMPLE_GRID_SIZE);
		const offsetX =
			((sampleX + 0.5) / AREA_LIGHT_SAMPLE_GRID_SIZE - 0.5) * width;
		const offsetY =
			((sampleY + 0.5) / AREA_LIGHT_SAMPLE_GRID_SIZE - 0.5) * height;
		const samplePoint = {
			x: center.x + right.x * offsetX + up.x * offsetY,
			y: center.y + right.y * offsetX + up.y * offsetY,
			z: center.z + right.z * offsetX + up.z * offsetY,
		};
		const lightVector = {
			x: samplePoint.x - surfacePos.x,
			y: samplePoint.y - surfacePos.y,
			z: samplePoint.z - surfacePos.z,
		};
		const distanceSq =
			lightVector.x * lightVector.x +
			lightVector.y * lightVector.y +
			lightVector.z * lightVector.z;
		const distance = Math.sqrt(Math.max(distanceSq, 1e-12));
		if (distance > range) continue;

		const normalized = {
			x: lightVector.x / distance,
			y: lightVector.y / distance,
			z: lightVector.z / distance,
		};
		const cosLight = Math.max(
			0,
			-(
				normal.x * normalized.x +
				normal.y * normalized.y +
				normal.z * normalized.z
			)
		);
		if (cosLight <= 0) continue;

		const projectedSolidAngle = rectangleProjectedSolidAngle(
			samplePoint,
			right,
			up,
			cellHalfWidth,
			cellHalfHeight,
			surfacePos
		);
		const sampleAttenuation =
			projectedSolidAngle * areaLightRangeAttenuation(distanceSq, range);
		accumulatedX += normalized.x * sampleAttenuation;
		accumulatedY += normalized.y * sampleAttenuation;
		accumulatedZ += normalized.z * sampleAttenuation;
		attenuation += sampleAttenuation;
	}

	if (attenuation <= 0) return null;

	const directionLength = Math.hypot(accumulatedX, accumulatedY, accumulatedZ);
	if (directionLength > 1e-12) {
		direction.x = accumulatedX / directionLength;
		direction.y = accumulatedY / directionLength;
		direction.z = accumulatedZ / directionLength;
	} else {
		direction.x = -normal.x;
		direction.y = -normal.y;
		direction.z = -normal.z;
	}

	return writeContribution(out, {
		type: "direct",
		color: light.color,
		intensity: light.intensity * attenuation,
		direction,
	});
}

function areaLightRangeAttenuation(distanceSq: number, range: number): number {
	const rangeSq = Math.max(range * range, 1e-12);
	const rangeFactor = distanceSq / rangeSq;
	const smoothFactor = Math.max(0, 1 - rangeFactor * rangeFactor);
	return smoothFactor * smoothFactor;
}

function rectangleProjectedSolidAngle(
	center: IVector3,
	right: IVector3,
	up: IVector3,
	halfWidth: number,
	halfHeight: number,
	worldPosition: IVector3
): number {
	const rightExtent = {
		x: right.x * halfWidth,
		y: right.y * halfWidth,
		z: right.z * halfWidth,
	};
	const upExtent = {
		x: up.x * halfHeight,
		y: up.y * halfHeight,
		z: up.z * halfHeight,
	};
	const p0 = {
		x: center.x - rightExtent.x - upExtent.x - worldPosition.x,
		y: center.y - rightExtent.y - upExtent.y - worldPosition.y,
		z: center.z - rightExtent.z - upExtent.z - worldPosition.z,
	};
	const p1 = {
		x: center.x + rightExtent.x - upExtent.x - worldPosition.x,
		y: center.y + rightExtent.y - upExtent.y - worldPosition.y,
		z: center.z + rightExtent.z - upExtent.z - worldPosition.z,
	};
	const p2 = {
		x: center.x + rightExtent.x + upExtent.x - worldPosition.x,
		y: center.y + rightExtent.y + upExtent.y - worldPosition.y,
		z: center.z + rightExtent.z + upExtent.z - worldPosition.z,
	};
	const p3 = {
		x: center.x - rightExtent.x + upExtent.x - worldPosition.x,
		y: center.y - rightExtent.y + upExtent.y - worldPosition.y,
		z: center.z - rightExtent.z + upExtent.z - worldPosition.z,
	};
	const solidAngle =
		sphericalTriangleSolidAngle(p0, p1, p2) +
		sphericalTriangleSolidAngle(p0, p2, p3);
	return Math.abs(solidAngle);
}

function sphericalTriangleSolidAngle(
	a: IVector3,
	b: IVector3,
	c: IVector3
): number {
	const an = normalizeForSolidAngle(a);
	const bn = normalizeForSolidAngle(b);
	const cn = normalizeForSolidAngle(c);
	const crossBC = {
		x: bn.y * cn.z - bn.z * cn.y,
		y: bn.z * cn.x - bn.x * cn.z,
		z: bn.x * cn.y - bn.y * cn.x,
	};
	const numerator = an.x * crossBC.x + an.y * crossBC.y + an.z * crossBC.z;
	const denominator =
		1 +
		(an.x * bn.x + an.y * bn.y + an.z * bn.z) +
		(bn.x * cn.x + bn.y * cn.y + bn.z * cn.z) +
		(cn.x * an.x + cn.y * an.y + cn.z * an.z);
	return 2 * Math.atan2(numerator, denominator);
}

function normalizeForSolidAngle(value: IVector3): IVector3 {
	const length = Math.hypot(value.x, value.y, value.z);
	if (length <= 1e-12) {
		return DEFAULT_DIRECTION;
	}
	return {
		x: value.x / length,
		y: value.y / length,
		z: value.z / length,
	};
}

function requireSurfacePosition(
	surface: SurfacePoint | null | undefined
): IVector3 {
	return surface?.position ?? ORIGIN;
}

function writeContribution(
	out: MutableLightContribution,
	input: LightContribution
): MutableLightContribution {
	out.type = input.type;
	out.color.r = input.color.r;
	out.color.g = input.color.g;
	out.color.b = input.color.b;
	out.intensity = input.intensity;

	if (input.direction) {
		out.direction.x = input.direction.x;
		out.direction.y = input.direction.y;
		out.direction.z = input.direction.z;
	}

	return out;
}

function toSrgb255(linear255: number): number {
	const linear01 = Math.max(0, linear255 / 255);
	return linearToSRGB(Math.min(1, linear01)) * 255;
}
