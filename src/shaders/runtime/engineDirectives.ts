import { ShaderRuntime } from "./ShaderRuntime";
import type {
	CompositeShaderSource,
	ShaderInjectionArgValue,
	ShaderInjectionScript,
	ShaderLanguage,
} from "./types";

interface EngineShaderDirectiveRequest {
	code: string;
	language: ShaderLanguage;
	sourcePath: string;
	label?: string;
}

const ENGINE_DIRECTIVE_RUNTIME = new ShaderRuntime({ mode: "warn" });
let _isConfigured = false;

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
		description: "Inject a shared luma() implementation with configurable profile.",
		run(args, context) {
			const profile = normalizeLumaProfile(args.profile);
			const clampInput = normalizeBooleanFlag(args.clamp, true);
			const weightsSymbol =
				profile === "bt601" ?
					"IGNIS_LUMA_WEIGHTS_BT601"
				:	"IGNIS_LUMA_WEIGHTS_BT709";
			if (context.language === "glsl") {
				return {
					functions: `float luma(vec3 color) {\n\treturn ignisLumaInternal(color, ${weightsSymbol}, ${clampInput ? "true" : "false"});\n}`,
					functionsAnchor: "afterUniforms",
				};
			}
			return {
				functions: `fn luma(color: vec3<f32>) -> f32 {\n\treturn ignisLumaInternal(color, ${weightsSymbol}, ${clampInput ? "true" : "false"});\n}`,
				functionsAnchor: "afterBindings",
			};
		},
	};
}

function configureEngineDirectiveRuntime(): void {
	if (_isConfigured) {
		return;
	}

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"glsl",
		"ignis/color/srgb.glsl",
		`vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

vec3 linearToSrgb(vec3 c) {
	vec3 a = c * 12.92;
	vec3 b = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
	return mix(b, a, lessThanEqual(c, vec3(0.0031308)));
}`,
		"runtime://ignis/includes/glsl/color/srgb.glsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"wgsl",
		"ignis/color/srgb.wgsl",
		`fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

fn srgbToLinear(color: vec3<f32>) -> vec3<f32> {
	return pow(max(color, vec3<f32>(0.0)), vec3<f32>(2.2));
}`,
		"runtime://ignis/includes/wgsl/color/srgb.wgsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"glsl",
		"ignis/postprocess/luma-weights.glsl",
		`#define IGNIS_LUMA_WEIGHTS_BT601 vec3(0.299, 0.587, 0.114)
#define IGNIS_LUMA_WEIGHTS_BT709 vec3(0.2126, 0.7152, 0.0722)`,
		"runtime://ignis/includes/glsl/postprocess/luma-weights.glsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"glsl",
		"ignis/postprocess/luma-common.glsl",
		`#include <ignis/postprocess/luma-weights>
float ignisLumaInternal(vec3 color, vec3 weights, bool clampInput) {
	vec3 sampleColor = clampInput ? max(color, vec3(0.0)) : color;
	return dot(sampleColor, weights);
}`,
		"runtime://ignis/includes/glsl/postprocess/luma-common.glsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"wgsl",
		"ignis/postprocess/luma-weights.wgsl",
		`const IGNIS_LUMA_WEIGHTS_BT601: vec3<f32> = vec3<f32>(0.299, 0.587, 0.114);
const IGNIS_LUMA_WEIGHTS_BT709: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);`,
		"runtime://ignis/includes/wgsl/postprocess/luma-weights.wgsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerIncludeModule(
		"wgsl",
		"ignis/postprocess/luma-common.wgsl",
		`#include <ignis/postprocess/luma-weights>
fn ignisLumaInternal(
	color: vec3<f32>,
	weights: vec3<f32>,
	clampInput: bool
) -> f32 {
	let sampleColor = select(color, max(color, vec3<f32>(0.0)), clampInput);
	return dot(sampleColor, weights);
}`,
		"runtime://ignis/includes/wgsl/postprocess/luma-common.wgsl"
	);

	ENGINE_DIRECTIVE_RUNTIME.registerInjectionScript(
		createPostProcessLumaInjectionScript()
	);
	_isConfigured = true;
}

export function preprocessEngineShaderDirectives(
	request: EngineShaderDirectiveRequest
): CompositeShaderSource {
	configureEngineDirectiveRuntime();
	const result = ENGINE_DIRECTIVE_RUNTIME.process({
		code: request.code,
		language: request.language,
		stage: "unknown",
		label: request.label ?? request.sourcePath,
		sourceKind: "unknown",
		directiveSourcePath: request.sourcePath,
	});
	return result.composite;
}
