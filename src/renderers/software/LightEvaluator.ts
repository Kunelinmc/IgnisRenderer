import { linearToSRGB } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import { SH } from "../../maths/SH";
import { Vector3 } from "../../maths/Vector3";
import type { IVector3 } from "../../maths/types";
import type { RGB } from "../../utils/Color";
import { LightType } from "../../lights/Light";
import type { AmbientLight } from "../../lights/AmbientLight";
import type { AreaLight } from "../../lights/AreaLight";
import type { DirectionalLight } from "../../lights/DirectionalLight";
import type { PointLight } from "../../lights/PointLight";
import type { LightProbe } from "../../lights/LightProbe";
import type { SpotLight } from "../../lights/SpotLight";
import {
	getDirectionalLightWorldDirection,
	getPointLightWorldPosition,
	getSpotLightInnerAngle,
	getSpotLightWorldDirection,
	getSpotLightWorldPosition,
} from "../../pipeline/LightTransforms";

type SceneLight =
	| AmbientLight
	| DirectionalLight
	| PointLight
	| SpotLight
	| LightProbe
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
	const direction = getDirectionalLightWorldDirection(light, out.direction);
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
	const lightPos = getPointLightWorldPosition(light);
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
	const lightPos = getSpotLightWorldPosition(light);
	const lightDir = getSpotLightWorldDirection(light);
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

	const outerCutoff = Math.cos(light.angle);
	const innerCutoff = Math.cos(getSpotLightInnerAngle(light));
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
		intensity: light.intensity,
	});
}

function evaluateAreaLight(
	light: AreaLight,
	surface: SurfacePoint,
	out: MutableLightContribution
): LightContribution | null {
	const surfacePos = requireSurfacePosition(surface);
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

	const projX = Vector3.dot(relPos, right);
	const projY = Vector3.dot(relPos, up);
	const halfW = light.width / 2;
	const halfH = light.height / 2;
	const clampedX = Math.max(-halfW, Math.min(halfW, projX));
	const clampedY = Math.max(-halfH, Math.min(halfH, projY));
	const closestPoint = Vector3.add(
		center,
		Vector3.add(Vector3.scale(right, clampedX), Vector3.scale(up, clampedY))
	);
	const lightVector = Vector3.sub(closestPoint, surfacePos);
	const distance = Vector3.length(lightVector);
	if (distance > light.range) return null;

	const direction = out.direction;
	const normalized = Vector3.normalize(lightVector);
	direction.x = normalized.x;
	direction.y = normalized.y;
	direction.z = normalized.z;

	const cosLight = Math.max(
		0,
		Vector3.dot(Vector3.scale(normal, -1), direction)
	);
	const distanceSq = distance * distance;
	const attenuation =
		((light.width * light.height) / 100) * (cosLight / (distanceSq + 1.0));

	return writeContribution(out, {
		type: "direct",
		color: light.color,
		intensity: light.intensity * attenuation,
		direction,
	});
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
