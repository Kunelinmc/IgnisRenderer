import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import type { DirectionalLight } from "../lights/DirectionalLight";
import type { PointLight } from "../lights/PointLight";
import type { SpotLight } from "../lights/SpotLight";

export function getDirectionalLightWorldDirection(
	light: DirectionalLight,
	out?: IVector3
): IVector3 {
	const direction = out ?? { x: 0, y: 0, z: 0 };
	const transformed = Vector3.normalize(
		Matrix4.transformDirection(light.worldMatrix, light.dir)
	);
	direction.x = transformed.x;
	direction.y = transformed.y;
	direction.z = transformed.z;
	return direction;
}

export function getPointLightWorldPosition(
	light: PointLight,
	out?: IVector3
): IVector3 {
	const position = out ?? { x: 0, y: 0, z: 0 };
	const transformed = Matrix4.transformPoint(light.worldMatrix, light.position);
	position.x = transformed.x;
	position.y = transformed.y;
	position.z = transformed.z;
	return position;
}

export function getSpotLightWorldPosition(
	light: SpotLight,
	out?: IVector3
): IVector3 {
	const position = out ?? { x: 0, y: 0, z: 0 };
	const transformed = Matrix4.transformPoint(light.worldMatrix, light.position);
	position.x = transformed.x;
	position.y = transformed.y;
	position.z = transformed.z;
	return position;
}

export function getSpotLightWorldDirection(
	light: SpotLight,
	out?: IVector3
): IVector3 {
	const direction = out ?? { x: 0, y: 0, z: 0 };
	const transformed = Vector3.normalize(
		Matrix4.transformDirection(light.worldMatrix, light.dir)
	);
	direction.x = transformed.x;
	direction.y = transformed.y;
	direction.z = transformed.z;
	return direction;
}

export function getSpotLightInnerAngle(light: SpotLight): number {
	return light.innerAngle ?? light.angle * (1 - light.penumbra);
}
