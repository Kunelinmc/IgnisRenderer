import type { AmbientLight } from "./AmbientLight";
import type { AreaLight } from "./AreaLight";
import type { DirectionalLight } from "./DirectionalLight";
import type { IrradianceProbeGrid } from "./IrradianceProbeGrid";
import type { LightProbe } from "./LightProbe";
import type { PointLight } from "./PointLight";
import type { ReflectionProbe } from "./ReflectionProbe";
import type { SpotLight } from "./SpotLight";

export type SceneLight =
	| AmbientLight
	| DirectionalLight
	| PointLight
	| SpotLight
	| LightProbe
	| IrradianceProbeGrid
	| ReflectionProbe
	| AreaLight;

export type ShadowCastingLight =
	| DirectionalLight
	| PointLight
	| SpotLight
	| AreaLight;
