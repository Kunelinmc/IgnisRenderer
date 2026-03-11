import { LightType } from "../../lights";
import type { LightProbe, SceneLight } from "../../lights";
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
	const envSpecularTexture = probe?.prefilteredMap ?? null;
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
		skyboxTexture: scene.skybox ?? envSpecularTexture,
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
