import { LightType } from "../../lights";
import type { LightProbe, SceneLight } from "../../lights";
import type { Texture } from "../../core/Texture";
import type { SHCoefficients } from "../../maths/types";
import { IBLBRDF } from "../../pipeline/IBLBRDF";
import type { PreparedScene } from "../../pipeline/types";

import type { WebGPUEnvironmentState, WebGPUWarning } from "./types";

export function collectWebGPUEnvironment(
	scene: PreparedScene,
	enableSH: boolean,
	shAmbientCoeffs: SHCoefficients | null
): WebGPUEnvironmentState {
	const warnings: WebGPUWarning[] = [];
	const probe = findFirstLightProbe(scene.lights);
	const sceneSkyboxTexture = resolveEnvironmentTexture(
		scene.skybox,
		"skybox",
		warnings
	);
	const envSpecularTexture = resolveEnvironmentTexture(
		probe?.prefilteredMap ?? null,
		"env-specular",
		warnings
	);
	const hasEnvSpecular = !!envSpecularTexture;

	const hasInputSHAmbient = hasNonZeroSH(shAmbientCoeffs);
	const hasSHAmbient = enableSH && hasInputSHAmbient;

	if (enableSH && !shAmbientCoeffs) {
		warnings.push({
			key: "webgpu-sh-missing-frame-context",
			message:
				"WebGPU SH was enabled without frame SH coefficients; falling back to non-SH ambient path",
		});
	}

	return {
		shAmbientCoeffs,
		enableSH,
		hasSHAmbient,
		skyboxTexture: sceneSkyboxTexture ?? envSpecularTexture,
		envSpecularTexture,
		brdfLUTTexture: hasEnvSpecular ? IBLBRDF.getLUT() : null,
		envSpecularMaxMipLevel:
			hasEnvSpecular ? Math.max(0, envSpecularTexture.mipmaps.length - 1) : 0,
		warnings,
	};
}

function findFirstLightProbe(lights: SceneLight[]): LightProbe | null {
	for (const light of lights) {
		if (light.type !== LightType.LightProbe) continue;
		const probe = light as LightProbe;
		if (probe.prefilteredMap) return probe;
	}
	return null;
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
	if (!texture) {
		return null;
	}

	if (texture.isLoadErrorFallback) {
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

	if (!isTextureReadyForEnvironment(texture)) {
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

	return texture;
}

function isTextureReadyForEnvironment(texture: Texture): boolean {
	if (
		!isFinitePositiveNumber(texture.width) ||
		!isFinitePositiveNumber(texture.height)
	) {
		return false;
	}

	return !!texture.data || texture.mipmaps.length > 0;
}

function isFinitePositiveNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}
