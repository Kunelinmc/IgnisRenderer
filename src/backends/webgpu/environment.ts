import type { Texture } from "../../core/Texture";
import type { LightProbe, SceneLight } from "../../lights";
import { SH } from "../../maths/SH";
import type { SHCoefficients } from "../../maths/types";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import {
	collectActiveLocalizedLightProbes,
	collectGlobalLightProbes,
} from "../../lights/runtime/lightProbeRuntime";
import { collectReflectionProbeEnvironment } from "../../lights/runtime/reflectionProbeRuntime";
import {
	ensureEnvironmentTextureEquirect,
	getEnvironmentMipLevelCount,
	isTextureReadyForEnvironment as isTextureReadyForEnvironmentShared,
} from "../../lights/runtime/environmentMapRuntime";
import type { PreparedScene } from "../../pipeline/types";

import {
	MAX_LOCAL_LIGHT_PROBES,
	MAX_REFLECTION_PROBES,
} from "../constants";
import type {
	WebGPUEnvironmentState,
	WebGPULocalLightProbeUniform,
	WebGPUReflectionProbeUniform,
	WebGPUWarning,
} from "./types";

const _tmpWebGPUReflectionProbeCameraWorldPosition = { x: 0, y: 0, z: 0 };

interface PreparedSceneEnvironmentLike {
	backgroundEnabled: boolean;
	lightingEnabled: boolean;
	backgroundTexture: Texture | null;
	iblTexture: Texture | null;
	backgroundStrength: number;
	backgroundTintLinear: { r: number; g: number; b: number };
	backgroundExposure: number;
}

export function collectWebGPUEnvironment(
	scene: PreparedScene,
	enableSH: boolean,
	shAmbientCoeffs: SHCoefficients | null
): WebGPUEnvironmentState {
	const warnings: WebGPUWarning[] = [];
	const environment = resolvePreparedSceneEnvironment(scene);
	const environmentBackgroundTexture =
		environment.backgroundEnabled ?
			resolveEnvironmentTexture(
				environment.backgroundTexture,
				"background",
				warnings
			)
		:	null;
	const environmentSpecularTexture =
		environment.lightingEnabled ?
			resolveEnvironmentTexture(
				environment.iblTexture,
				"env-specular",
				warnings
			)
		:	null;
	const reflectionProbeCameraWorldPosition =
		typeof scene.camera?.getWorldPosition === "function" ?
			scene.camera.getWorldPosition(_tmpWebGPUReflectionProbeCameraWorldPosition)
		:	null;
	const localLightProbes = enableSH ?
			collectActiveLocalizedLightProbes(
				scene.lights,
				MAX_LOCAL_LIGHT_PROBES,
				reflectionProbeCameraWorldPosition
			)
		:	[];
	const reflectionEnvironment = collectReflectionProbeEnvironment(
		scene.lights,
		MAX_REFLECTION_PROBES,
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
				"WebGPU reflection probes are active but atlas build failed or is not ready; using environment IBL texture only.",
		});
	}
	if (!envSpecularTexture) {
		envSpecularTexture = environmentSpecularTexture;
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
		enableSH &&
		(hasInputSHAmbient || localLightProbeCount > 0);

	return {
		shAmbientCoeffs: resolvedSHAmbientCoeffs,
		enableSH,
		hasSHAmbient,
		environmentTexture: environmentBackgroundTexture,
		envSpecularTexture,
		localLightProbeCount,
		localLightProbes: localizedProbeUniforms,
		reflectionProbeCount,
		reflectionProbes,
		brdfLUTTexture: IBLBRDF.getLUT(),
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
	const globalLightProbes = collectGlobalLightProbes(lights);

	for (const light of globalLightProbes) {
		const probeSH = light.sh;
		const coeffCount = Math.min(ambientProbeSH.length, probeSH.length);
		for (let i = 0; i < coeffCount; i++) {
			ambientProbeSH[i].r += probeSH[i].r;
			ambientProbeSH[i].g += probeSH[i].g;
			ambientProbeSH[i].b += probeSH[i].b;
		}
	}

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
		sh: cloneSHCoefficients(probe.sh),
	};
}

function cloneSHCoefficients(source: SHCoefficients): SHCoefficients {
	const result = SH.empty();
	for (let i = 0; i < result.length; i++) {
		result[i].r = source[i].r;
		result[i].g = source[i].g;
		result[i].b = source[i].b;
	}
	return result;
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

type EnvironmentTextureSlot = "background" | "env-specular";

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
				slot === "background" ?
					"webgpu-environment-background-load-error-fallback"
				:	"webgpu-env-specular-load-error-fallback",
			message:
				slot === "background" ?
					"WebGPU environment background texture resolved to a load-error fallback; skipping environment background rendering."
				:	"WebGPU environment specular texture resolved to a load-error fallback; skipping IBL specular.",
		});
		return null;
	}

	if (!isTextureReadyForEnvironmentShared(normalizedTexture)) {
		warnings.push({
			key:
				slot === "background" ?
					"webgpu-environment-background-texture-not-ready"
				:	"webgpu-env-specular-texture-not-ready",
			message:
				slot === "background" ?
					"WebGPU environment background texture is not ready (missing pixels or invalid dimensions); skipping environment background rendering for this frame."
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

function resolvePreparedSceneEnvironment(
	scene: PreparedScene
): PreparedSceneEnvironmentLike {
	const rawEnvironment = (scene as { environment?: unknown }).environment;
	if (!rawEnvironment || typeof rawEnvironment !== "object") {
		return {
			backgroundEnabled: true,
			lightingEnabled: true,
			backgroundTexture: null,
			iblTexture: null,
			backgroundStrength: 1,
			backgroundTintLinear: { r: 1, g: 1, b: 1 },
			backgroundExposure: 1,
		};
	}
	const environment = rawEnvironment as Partial<PreparedSceneEnvironmentLike>;
	return {
		backgroundEnabled: environment.backgroundEnabled ?? true,
		lightingEnabled: environment.lightingEnabled ?? true,
		backgroundTexture: (environment.backgroundTexture ?? null) as Texture | null,
		iblTexture: (environment.iblTexture ?? null) as Texture | null,
		backgroundStrength:
			typeof environment.backgroundStrength === "number" ?
				environment.backgroundStrength
			:	1,
		backgroundTintLinear: {
			r:
				typeof environment.backgroundTintLinear?.r === "number" ?
					environment.backgroundTintLinear.r
				:	1,
			g:
				typeof environment.backgroundTintLinear?.g === "number" ?
					environment.backgroundTintLinear.g
				:	1,
			b:
				typeof environment.backgroundTintLinear?.b === "number" ?
					environment.backgroundTintLinear.b
				:	1,
		},
		backgroundExposure:
			typeof environment.backgroundExposure === "number" ?
				environment.backgroundExposure
			:	1,
	};
}
