import type { Texture } from "../../core/Texture";
import {
	LightType,
	type LightProbe,
	type SceneLight,
} from "../../lights";
import { PBR_AMBIENT_FALLBACK_LINEAR } from "../../lights/constants";
import { sRGBToLinear } from "../../maths/Common";
import { SH } from "../../maths/SH";
import type { SHCoefficients } from "../../maths/types";
import { IBLBRDF } from "../../pipeline/IBLBRDF";
import {
	collectActiveLocalizedLightProbes,
	collectGlobalLightProbes,
} from "../../pipeline/lightProbeRuntime";
import { collectReflectionProbeEnvironment } from "../../pipeline/reflectionProbeRuntime";
import {
	ensureEnvironmentTextureEquirect,
	getEnvironmentMipLevelCount,
	isTextureReadyForEnvironment as isTextureReadyForEnvironmentShared,
} from "../../pipeline/environmentMapRuntime";
import type { PreparedScene } from "../../pipeline/types";

import {
	WEBGPU_MAX_LOCAL_LIGHT_PROBES,
	WEBGPU_MAX_REFLECTION_PROBES,
} from "./constants";
import type {
	WebGPUEnvironmentState,
	WebGPULocalLightProbeUniform,
	WebGPUReflectionProbeUniform,
	WebGPUWarning,
} from "./types";

const SH_DC_IRRADIANCE_SCALE = Math.PI * 0.282095;
const _tmpWebGPUReflectionProbeCameraWorldPosition = { x: 0, y: 0, z: 0 };

export function collectWebGPUEnvironment(
	scene: PreparedScene,
	enableSH: boolean,
	shAmbientCoeffs: SHCoefficients | null
): WebGPUEnvironmentState {
	const warnings: WebGPUWarning[] = [];
	const allowSkyboxSpecularFallback =
		scene.allowSkyboxSpecularFallback !== false;
	const sceneSkyboxTexture = resolveEnvironmentTexture(
		scene.skybox,
		"skybox",
		warnings
	);
	const reflectionProbeCameraWorldPosition =
		typeof scene.camera?.getWorldPosition === "function" ?
			scene.camera.getWorldPosition(_tmpWebGPUReflectionProbeCameraWorldPosition)
		:	null;
	const localLightProbes = enableSH ?
			collectActiveLocalizedLightProbes(
				scene.lights,
				WEBGPU_MAX_LOCAL_LIGHT_PROBES,
				reflectionProbeCameraWorldPosition
			)
		:	[];
	const reflectionEnvironment = collectReflectionProbeEnvironment(
		scene.lights,
		WEBGPU_MAX_REFLECTION_PROBES,
		reflectionProbeCameraWorldPosition
	);
	let reflectionProbeCount = 0;
	let reflectionProbes: WebGPUReflectionProbeUniform[] = [];
	let envSpecularTexture = resolveEnvironmentTexture(
		reflectionEnvironment.atlas,
		"env-specular",
		warnings
	);
	if (envSpecularTexture) {
		reflectionProbes = reflectionEnvironment.probes.map((probe, index) => {
			const cache = probe.getRuntimeCache();
			return {
				id: probe.id,
				worldToProbeMatrix: cache.worldToProbeMatrix.clone(),
				probeToWorldMatrix: cache.probeToWorldMatrix.clone(),
				invHalfExtents: [
					cache.invHalfExtents.x,
					cache.invHalfExtents.y,
					cache.invHalfExtents.z,
				],
				radiusInv: cache.radiusInv,
				captureWorldPosition: [
					cache.captureWorldPosition.x,
					cache.captureWorldPosition.y,
					cache.captureWorldPosition.z,
				],
				shape: probe.shape === "box" ? 1 : 0,
				parallaxMode: mapParallaxModeCode(probe.parallaxMode),
				blendDistance: cache.effectiveBlendDistance,
				blendExponent: cache.blendExponent,
				layer: index,
			};
		});
		reflectionProbeCount = reflectionProbes.length;
	} else if (reflectionEnvironment.probes.length > 0) {
		warnings.push({
			key: "webgpu-reflection-probe-atlas-missing",
			message:
				"WebGPU reflection probes are active but atlas build failed or is not ready; falling back to skybox environment specular.",
		});
	}
	if (!envSpecularTexture && allowSkyboxSpecularFallback) {
		envSpecularTexture = sceneSkyboxTexture;
	}
	const hasEnvSpecular = !!envSpecularTexture;
	const resolvedSHAmbientCoeffs = resolveSHAmbientCoeffs(
		scene.lights,
		enableSH,
		shAmbientCoeffs
	);
	const localizedProbeUniforms = localLightProbes.map((probe) =>
		createWebGPULocalLightProbeUniform(probe)
	);
	const localLightProbeCount = localizedProbeUniforms.length;

	const hasInputSHAmbient = hasNonZeroSH(resolvedSHAmbientCoeffs);
	const hasSHAmbient =
		enableSH && (hasInputSHAmbient || localLightProbeCount > 0);

	return {
		shAmbientCoeffs: resolvedSHAmbientCoeffs,
		enableSH,
		hasSHAmbient,
		skyboxTexture: sceneSkyboxTexture,
		envSpecularTexture,
		localLightProbeCount,
		localLightProbes: localizedProbeUniforms,
		reflectionProbeCount,
		reflectionProbes,
		brdfLUTTexture: hasEnvSpecular ? IBLBRDF.getLUT() : null,
		envSpecularMaxMipLevel:
			hasEnvSpecular ? Math.max(0, getEnvironmentMipLevelCount(envSpecularTexture) - 1) : 0,
		warnings,
	};
}

function resolveSHAmbientCoeffs(
	lights: SceneLight[],
	enableSH: boolean,
	shAmbientCoeffs: SHCoefficients | null
): SHCoefficients | null {
	if (!enableSH) return shAmbientCoeffs;
	if (shAmbientCoeffs) return shAmbientCoeffs;
	return synthesizeSHAmbientCoeffsFromLights(lights);
}

function synthesizeSHAmbientCoeffsFromLights(
	lights: SceneLight[]
): SHCoefficients {
	const ambientProbeSH = SH.empty();
	let ambientR = 0;
	let ambientG = 0;
	let ambientB = 0;
	let hasAmbient = false;
	const globalLightProbes = collectGlobalLightProbes(lights);

	for (const light of lights) {
		if (light.type === LightType.Ambient) {
			const color = light.color ?? { r: 255, g: 255, b: 255 };
			const intensity = light.intensity ?? 1;
			ambientR += sRGBToLinear(color.r / 255) * 255 * intensity;
			ambientG += sRGBToLinear(color.g / 255) * 255 * intensity;
			ambientB += sRGBToLinear(color.b / 255) * 255 * intensity;
			hasAmbient = true;
			continue;
		}

	}

	for (const light of globalLightProbes) {
		const probeSH = light.sh;
		const intensity = light.intensity ?? 1;
		const coeffCount = Math.min(ambientProbeSH.length, probeSH.length);
		for (let i = 0; i < coeffCount; i++) {
			ambientProbeSH[i].r += probeSH[i].r * intensity;
			ambientProbeSH[i].g += probeSH[i].g * intensity;
			ambientProbeSH[i].b += probeSH[i].b * intensity;
		}
	}

	if (
		!hasAmbient &&
		ambientProbeSH[0].r === 0 &&
		ambientProbeSH[0].g === 0 &&
		ambientProbeSH[0].b === 0
	) {
		const fallbackLinear = PBR_AMBIENT_FALLBACK_LINEAR * 255;
		ambientR = fallbackLinear;
		ambientG = fallbackLinear;
		ambientB = fallbackLinear;
	}

	ambientProbeSH[0].r += ambientR / SH_DC_IRRADIANCE_SCALE;
	ambientProbeSH[0].g += ambientG / SH_DC_IRRADIANCE_SCALE;
	ambientProbeSH[0].b += ambientB / SH_DC_IRRADIANCE_SCALE;
	return ambientProbeSH;
}

function createWebGPULocalLightProbeUniform(
	probe: LightProbe
): WebGPULocalLightProbeUniform {
	const cache = probe.getRuntimeCache();
	return {
		id: probe.id,
		worldToProbeMatrix: cache.worldToProbeMatrix.clone(),
		invHalfExtents: [
			cache.invHalfExtents.x,
			cache.invHalfExtents.y,
			cache.invHalfExtents.z,
		],
		radiusInv: cache.radiusInv,
		shape: probe.shape === "box" ? 1 : 0,
		blendDistance: cache.effectiveBlendDistance,
		priority: cache.priority,
		sh: probe.sh.map((coefficient) => ({
			r: coefficient.r,
			g: coefficient.g,
			b: coefficient.b,
		})),
	};
}

function hasNonZeroSH(coeffs: SHCoefficients | null): boolean {
	if (!coeffs) return false;
	for (const coefficient of coeffs) {
		if (coefficient.r !== 0 || coefficient.g !== 0 || coefficient.b !== 0) {
			return true;
		}
	}
	return false;
}

type EnvironmentTextureSlot = "skybox" | "env-specular";

function resolveEnvironmentTexture(
	texture: Texture | null | undefined,
	slot: EnvironmentTextureSlot,
	warnings: WebGPUWarning[]
): Texture | null {
	const normalizedTexture = ensureEnvironmentTextureEquirect(texture);
	if (!normalizedTexture) {
		return null;
	}

	if (normalizedTexture.isLoadErrorFallback) {
		warnings.push({
			key:
				slot === "skybox" ?
					"webgpu-skybox-load-error-fallback"
				:	"webgpu-env-specular-load-error-fallback",
			message:
				slot === "skybox" ?
					"WebGPU skybox texture resolved to a load-error fallback; skipping skybox rendering."
				:	"WebGPU environment specular texture resolved to a load-error fallback; skipping IBL specular.",
		});
		return null;
	}

	if (!isTextureReadyForEnvironmentShared(normalizedTexture)) {
		warnings.push({
			key:
				slot === "skybox" ?
					"webgpu-skybox-texture-not-ready"
				:	"webgpu-env-specular-texture-not-ready",
			message:
				slot === "skybox" ?
					"WebGPU skybox texture is not ready (missing pixels or invalid dimensions); skipping skybox rendering for this frame."
				:	"WebGPU environment specular texture is not ready (missing pixels or invalid dimensions); skipping IBL specular for this frame.",
		});
		return null;
	}

	return normalizedTexture;
}

function mapParallaxModeCode(mode: string): 0 | 1 | 2 {
	if (mode === "box") return 1;
	if (mode === "sphere") return 2;
	return 0;
}
