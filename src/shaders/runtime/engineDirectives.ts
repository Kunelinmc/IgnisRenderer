import type {
	ShaderDirectiveProfile,
	ShaderDirectiveProfileRegistry,
	ShaderInjectionArgValue,
	ShaderInjectionScript,
} from "./types";

const WEBGPU_PROFILE_ID = "webgpu/v1";
const WEBGL_PROFILE_ID = "webgl/v1";
const SOFTWARE_PROFILE_ID = "software/v1";
const PROFILE_REVISION = 1;
const MIGRATION_HINT =
	" Migration hint: use ShaderBackendCompileStage with explicit webgpu/webgl/software directive profiles.";

function normalizeLumaProfile(
	value: ShaderInjectionArgValue | undefined
): "bt601" | "bt709" {
	if (typeof value === "string") {
		const normalized = value.trim().toLowerCase();
		if (normalized === "bt601" || normalized === "601") {
			return "bt601";
		}
	}
	return "bt709";
}

function normalizeBooleanFlag(
	value: ShaderInjectionArgValue | undefined,
	fallback: boolean
): boolean {
	if (typeof value === "boolean") {
		return value;
	}
	if (typeof value === "number") {
		return value !== 0;
	}
	if (typeof value === "string") {
		const normalized = value.trim().toLowerCase();
		if (
			normalized === "false" ||
			normalized === "0" ||
			normalized === "no" ||
			normalized === "off"
		) {
			return false;
		}
		if (
			normalized === "true" ||
			normalized === "1" ||
			normalized === "yes" ||
			normalized === "on"
		) {
			return true;
		}
	}
	return fallback;
}

function createPostProcessLumaInjectionScript(): ShaderInjectionScript {
	return {
		id: "ignis/postprocess/luma",
		description:
			"Inject a shared luma() implementation with configurable profile.",
		run(args, context) {
			const profile = normalizeLumaProfile(args.profile);
			const clampInput = normalizeBooleanFlag(args.clamp, true);
			const weightsExpression =
				context.language === "glsl" ?
					(
						profile === "bt601" ?
							"vec3(0.299, 0.587, 0.114)"
						:	"vec3(0.2126, 0.7152, 0.0722)"
					)
				:	(
						profile === "bt601" ?
							"vec3<f32>(0.299, 0.587, 0.114)"
						:	"vec3<f32>(0.2126, 0.7152, 0.0722)"
					);
			if (context.language === "glsl") {
				return {
					functions: `float luma(vec3 color) {\n\treturn ignisLumaInternal(color, ${weightsExpression}, ${clampInput ? "true" : "false"});\n}`,
					functionsAnchor: "afterUniforms",
				};
			}
			return {
				functions: `fn luma(color: vec3<f32>) -> f32 {\n\treturn ignisLumaInternal(color, ${weightsExpression}, ${clampInput ? "true" : "false"});\n}`,
				functionsAnchor: "afterBindings",
			};
		},
	};
}

function createWebGPUProfile(): ShaderDirectiveProfile {
	return {
		id: WEBGPU_PROFILE_ID,
		backend: "webgpu",
		revision: PROFILE_REVISION,
		includeModules: [
			{
				language: "wgsl",
				id: "ignis/color/srgb.wgsl",
				code: `fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn srgbToLinear(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(2.2));
}`,
				sourcePath: "runtime://ignis/includes/wgsl/color/srgb.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/postprocess/luma-weights.wgsl",
				code: `const IGNIS_LUMA_WEIGHTS_BT601: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);
const IGNIS_LUMA_WEIGHTS_BT709: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);`,
				sourcePath: "runtime://ignis/includes/wgsl/postprocess/luma-weights.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/postprocess/luma-common.wgsl",
				code: `#include <ignis/postprocess/luma-weights>
fn ignisLumaInternal(
	color: vec3<f32>,
	weights: vec3<f32>,
	clampInput: bool
) -> f32 {
	let sampleColor = select(color, max(color, vec3<f32>(0.0)), clampInput);
	return dot(sampleColor, weights);
}`,
				sourcePath: "runtime://ignis/includes/wgsl/postprocess/luma-common.wgsl",
			},
		],
		injectionScripts: [createPostProcessLumaInjectionScript()],
	};
}

function createWebGLProfile(): ShaderDirectiveProfile {
	return {
		id: WEBGL_PROFILE_ID,
		backend: "webgl",
		revision: PROFILE_REVISION,
		includeModules: [
			{
				language: "glsl",
				id: "ignis/color/srgb.glsl",
				code: `vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

vec3 linearToSrgb(vec3 c) {
	vec3 a = c * 12.92;
	vec3 b = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
	return mix(b, a, lessThanEqual(c, vec3(0.0031308)));
}`,
				sourcePath: "runtime://ignis/includes/glsl/color/srgb.glsl",
			},
			{
				language: "glsl",
				id: "ignis/postprocess/luma-weights.glsl",
				code: `#define IGNIS_LUMA_WEIGHTS_BT601 vec3(0.299, 0.587, 0.114)
#define IGNIS_LUMA_WEIGHTS_BT709 vec3(0.2126, 0.7152, 0.0722)`,
				sourcePath: "runtime://ignis/includes/glsl/postprocess/luma-weights.glsl",
			},
			{
				language: "glsl",
				id: "ignis/postprocess/luma-common.glsl",
				code: `#include <ignis/postprocess/luma-weights>
float ignisLumaInternal(vec3 color, vec3 weights, bool clampInput) {
	vec3 sampleColor = clampInput ? max(color, vec3(0.0)) : color;
	return dot(sampleColor, weights);
}`,
				sourcePath: "runtime://ignis/includes/glsl/postprocess/luma-common.glsl",
			},
		],
		injectionScripts: [createPostProcessLumaInjectionScript()],
	};
}

function createSoftwareProfile(): ShaderDirectiveProfile {
	return {
		id: SOFTWARE_PROFILE_ID,
		backend: "software",
		revision: PROFILE_REVISION,
		includeModules: [],
		injectionScripts: [],
	};
}

export function createDefaultShaderDirectiveProfileRegistry():
	ShaderDirectiveProfileRegistry {
	return {
		webgpu: createWebGPUProfile(),
		webgl: createWebGLProfile(),
		software: createSoftwareProfile(),
	};
}

export const DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY =
	createDefaultShaderDirectiveProfileRegistry();

export function assertShaderDirectiveProfileRegistryComplete(
	registry: Partial<ShaderDirectiveProfileRegistry> | null | undefined
): asserts registry is ShaderDirectiveProfileRegistry {
	const normalized = registry ?? {};
	for (const backend of ["webgpu", "webgl", "software"] as const) {
		const profile = normalized[backend];
		if (!profile) {
			throw new Error(
				`Missing shader directive profile for backend "${backend}".${MIGRATION_HINT}`
			);
		}
		if (profile.backend !== backend) {
			throw new Error(
				`Shader directive profile "${profile.id}" has mismatched backend "${profile.backend}" (expected "${backend}").${MIGRATION_HINT}`
			);
		}
	}
}
