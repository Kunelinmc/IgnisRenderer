import type {
	ShaderDirectiveProfile,
	ShaderDirectiveProfileRegistry,
	ShaderInjectionArgValue,
	ShaderInjectionScript,
	ShaderLanguage,
} from "./types";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_QUALITY,
	MAX_AREA_LIGHTS,
	MAX_CLUSTER_LIGHTS_PER_FRAGMENT,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
	VOLUMETRIC_SIGMA_T_SCALE,
} from "../../renderers/constants";
import {
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../renderers/webgpu/constants";

const WEBGPU_PROFILE_ID = "webgpu/v1";
const WEBGL_PROFILE_ID = "webgl/v1";
const SOFTWARE_PROFILE_ID = "software/v1";
const PROFILE_REVISION = 12;
const MATERIAL_TEXTURE_SLOT_COUNT = 14;
const MATERIAL_SHADER_UNIFORM_BINDING = WEBGPU_MODEL_BINDING_SHADER_UNIFORMS;
const MIGRATION_HINT =
	" Migration hint: use ShaderBackendCompileStage with explicit webgpu/webgl/software directive profiles.";
type LumaProfile = "bt601" | "bt709";
type VecDimension = 2 | 3 | 4;
type ShaderMaterialUniformType =
	| "f32"
	| "i32"
	| "u32"
	| "vec2f"
	| "vec3f"
	| "vec4f"
	| "vec2i"
	| "vec3i"
	| "vec4i"
	| "vec2u"
	| "vec3u"
	| "vec4u"
	| "mat4x4f";
interface ShaderMaterialUniformField {
	wgslField: string;
	type: ShaderMaterialUniformType;
	webglUniform: string;
}
const FXAA_EDGE_THRESHOLD_MIN_LITERAL = toShaderFloat(
	FXAA_EDGE_THRESHOLD_MIN
);
const FXAA_QUALITY_WGSL_LITERAL = FXAA_QUALITY.map(toShaderFloat).join(", ");
const FXAA_QUALITY_GLSL_LITERAL = FXAA_QUALITY.map(toShaderFloat).join(", ");
const VOLUMETRIC_SIGMA_T_SCALE_LITERAL = toShaderFloat(
	VOLUMETRIC_SIGMA_T_SCALE
);
const WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT =
	MAX_LOCAL_LIGHT_PROBES * WEBGPU_SH_COEFFICIENT_COUNT;
const WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT =
	MAX_DIRECTIONAL_LIGHTS * 2;
const WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT = MAX_POINT_LIGHTS * 2;
const WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT = MAX_SPOT_LIGHTS * 3;

function toShaderFloat(value: number): string {
	if (!Number.isFinite(value)) {
		return "0.0";
	}
	const text = `${value}`;
	return text.includes(".") ? text : `${text}.0`;
}

function getVecConstructor(
	lang: ShaderLanguage,
	dimension: VecDimension,
	values: string
): string {
	const name = `vec${dimension}`;
	return lang === "glsl" ? `${name}(${values})` : `${name}<f32>(${values})`;
}

function getLumaWeightsExpression(lang: ShaderLanguage, profile: LumaProfile): string {
	const weights =
		profile === "bt601" ?
			"0.299, 0.587, 0.114"
		:	"0.2126, 0.7152, 0.0722";
	return getVecConstructor(lang, 3, weights);
}

function getDecodeExpression(lang: ShaderLanguage, linear: boolean): string {
	if (linear) {
		return "sampled";
	}
	const decodedRgb =
		lang === "glsl" ?
			`mix(pow((sampled.rgb + vec3(0.055)) / vec3(1.055), vec3(2.4)), sampled.rgb / vec3(12.92), lessThanEqual(sampled.rgb, vec3(0.04045)))`
		:	`select(pow((sampled.rgb + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4)), sampled.rgb / vec3<f32>(12.92), sampled.rgb <= vec3<f32>(0.04045))`;
	return getVecConstructor(lang, 4, `${decodedRgb}, sampled.a`);
}

function normalizeLumaProfile(
	value: ShaderInjectionArgValue | undefined
): LumaProfile {
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

function normalizeTextureSlot(value: ShaderInjectionArgValue | undefined): number {
	const numeric =
		typeof value === "number" ? value
		: typeof value === "string" ? Number(value)
		:	NaN;
	if (Number.isFinite(numeric)) {
		const resolved = Math.floor(numeric);
		if (resolved >= 0 && resolved < MATERIAL_TEXTURE_SLOT_COUNT) {
			return resolved;
		}
	}
	return 0;
}

function normalizeTextureUVSet(
	value: ShaderInjectionArgValue | undefined
): 0 | 1 | 2 | 3 {
	const numeric =
		typeof value === "number" ? value
		: typeof value === "string" ? Number(value)
		:	NaN;
	if (!Number.isFinite(numeric)) {
		return 0;
	}
	return Math.max(0, Math.min(3, Math.floor(numeric))) as 0 | 1 | 2 | 3;
}

function normalizeIdentifierToken(
	value: ShaderInjectionArgValue | undefined,
	fallback: string
): string {
	const raw =
		typeof value === "string" && value.trim().length > 0 ?
			value.trim()
		:	fallback;
	const sanitized = raw.replace(/[^A-Za-z0-9_]/g, "_");
	if (/^[A-Za-z_]/.test(sanitized)) {
		return sanitized;
	}
	return `x_${sanitized}`;
}

function hasGLSLSamplerUniform(source: string, uniformName: string): boolean {
	const pattern = new RegExp(
		`\\buniform\\s+sampler2D\\s+${uniformName}\\b`,
		"m"
	);
	return pattern.test(source);
}

function hasGLSLUniform(source: string, uniformName: string): boolean {
	const pattern = new RegExp(
		`\\buniform\\s+[A-Za-z_][A-Za-z0-9_]*(?:\\s*<[^>]+>)?\\s+${uniformName}\\b`,
		"m"
	);
	return pattern.test(source);
}

function parseMaterialUniformFields(
	value: ShaderInjectionArgValue | undefined
): ShaderMaterialUniformField[] {
	if (typeof value !== "string" || value.trim().length <= 0) {
		return [];
	}
	const fields: ShaderMaterialUniformField[] = [];
	for (const rawField of value.split(";")) {
		const trimmed = rawField.trim();
		if (trimmed.length <= 0) {
			continue;
		}
		const [rawWgslField, rawType, rawWebglUniform] = trimmed.split(":");
		const wgslField = normalizeIdentifierToken(rawWgslField, "field");
		const webglUniform = normalizeIdentifierToken(rawWebglUniform, "uShaderUniform");
		const type = normalizeMaterialUniformType(rawType);
		fields.push({
			wgslField,
			type,
			webglUniform,
		});
	}
	return fields;
}

function normalizeMaterialUniformType(
	value: string | undefined
): ShaderMaterialUniformType {
	switch (value) {
		case "f32":
		case "i32":
		case "u32":
		case "vec2f":
		case "vec3f":
		case "vec4f":
		case "vec2i":
		case "vec3i":
		case "vec4i":
		case "vec2u":
		case "vec3u":
		case "vec4u":
		case "mat4x4f":
			return value;
		default:
			return "f32";
	}
}

function getWGSLUniformType(type: ShaderMaterialUniformType): string {
	switch (type) {
		case "i32":
			return "i32";
		case "u32":
			return "u32";
		case "vec2f":
			return "vec2<f32>";
		case "vec3f":
			return "vec3<f32>";
		case "vec4f":
			return "vec4<f32>";
		case "vec2i":
			return "vec2<i32>";
		case "vec3i":
			return "vec3<i32>";
		case "vec4i":
			return "vec4<i32>";
		case "vec2u":
			return "vec2<u32>";
		case "vec3u":
			return "vec3<u32>";
		case "vec4u":
			return "vec4<u32>";
		case "mat4x4f":
			return "mat4x4<f32>";
		case "f32":
		default:
			return "f32";
	}
}

function getGLSLUniformType(type: ShaderMaterialUniformType): string {
	switch (type) {
		case "i32":
			return "int";
		case "u32":
			return "uint";
		case "vec2f":
			return "vec2";
		case "vec3f":
			return "vec3";
		case "vec4f":
			return "vec4";
		case "vec2i":
			return "ivec2";
		case "vec3i":
			return "ivec3";
		case "vec4i":
			return "ivec4";
		case "vec2u":
			return "uvec2";
		case "vec3u":
			return "uvec3";
		case "vec4u":
			return "uvec4";
		case "mat4x4f":
			return "mat4";
		case "f32":
		default:
			return "float";
	}
}

function findWGSLBindingVariableName(
	source: string,
	binding: number
): string | null {
	const patterns = [
		new RegExp(
			`@group\\s*\\(\\s*1\\s*\\)\\s*@binding\\s*\\(\\s*${binding}\\s*\\)\\s*` +
				`(?:@[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\([^)]*\\))?\\s*)*` +
				`var(?:<[^>]+>)?\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:`,
			"m"
		),
		new RegExp(
			`@binding\\s*\\(\\s*${binding}\\s*\\)\\s*@group\\s*\\(\\s*1\\s*\\)\\s*` +
				`(?:@[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\([^)]*\\))?\\s*)*` +
				`var(?:<[^>]+>)?\\s+([A-Za-z_][A-Za-z0-9_]*)\\s*:`,
			"m"
		),
	];
	for (const pattern of patterns) {
		const match = pattern.exec(source);
		if (match?.[1]) {
			return match[1];
		}
	}
	return null;
}

function createPostProcessLumaInjectionScript(): ShaderInjectionScript {
	return {
		id: "ignis/postprocess/luma",
		description:
			"Inject a shared luma() implementation with configurable profile.",
		run(args, context) {
			const profile = normalizeLumaProfile(args.profile);
			const clampInput = normalizeBooleanFlag(args.clamp, true);
			const weightsExpression = getLumaWeightsExpression(
				context.language,
				profile
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

function createMaterialTextureBindingInjectionScript(): ShaderInjectionScript {
	return {
		id: "ignis/material/texture-binding",
		description:
			"Inject per-material texture helper declarations for custom shader materials.",
		run(args, context) {
			if (
				context.sourceKind !== "custom-material" ||
				context.stage !== "fragment"
			) {
				return null;
			}

			const nameToken = normalizeIdentifierToken(args.name, "texture");
			const uniformName = normalizeIdentifierToken(
				args.uniform,
				`uShaderTex_${nameToken}`
			);
			const slot = normalizeTextureSlot(args.slot);
			const uvSet = normalizeTextureUVSet(args.uv);
			const linear = normalizeBooleanFlag(args.linear, false);
			const symbolToken = nameToken.toUpperCase();
			const fnName = `ignisSampleTexture_${nameToken}`;
			const slotConst = `IGNIS_TEXTURE_SLOT_${symbolToken}`;
			const uvConst = `IGNIS_TEXTURE_UVSET_${symbolToken}`;
			const linearConst = `IGNIS_TEXTURE_LINEAR_${symbolToken}`;
			const decodeExpression = getDecodeExpression(context.language, linear);

			if (context.language === "glsl") {
				const headerBlocks: string[] = [];
				if (!hasGLSLSamplerUniform(context.source, uniformName)) {
					headerBlocks.push(`uniform sampler2D ${uniformName};`);
				}
				headerBlocks.push(
					`const int ${slotConst} = ${slot};`,
					`const int ${uvConst} = ${uvSet};`,
					`const bool ${linearConst} = ${linear ? "true" : "false"};`
				);
				return {
					header: headerBlocks.join("\n"),
					functions:
						`vec4 ${fnName}(vec2 uv0, vec2 uv1, vec2 uv2, vec2 uv3) {\n` +
						`\tvec2 uv = uv0;\n` +
						`\tif (${uvConst} == 1) uv = uv1;\n` +
						`\telse if (${uvConst} == 2) uv = uv2;\n` +
						`\telse if (${uvConst} >= 3) uv = uv3;\n` +
						`\tvec4 sampled = texture(${uniformName}, uv);\n` +
						`\treturn ${decodeExpression};\n` +
						`}`,
					symbols: [fnName, slotConst, uvConst, linearConst, uniformName],
					headerAnchor: "afterUniforms",
					functionsAnchor: "beforeEntryPoint",
				};
			}

			const textureBindingIndex = slot * 2 + 1;
			const samplerBindingIndex = textureBindingIndex + 1;
			const existingTextureName = findWGSLBindingVariableName(
				context.source,
				textureBindingIndex
			);
			const existingSamplerName = findWGSLBindingVariableName(
				context.source,
				samplerBindingIndex
			);
			const textureName = existingTextureName ?? `ignisShaderTexture_${nameToken}`;
			const samplerName = existingSamplerName ?? `ignisShaderSampler_${nameToken}`;
			const headerBlocks: string[] = [];
			if (!existingTextureName) {
				headerBlocks.push(
					`@group(1) @binding(${textureBindingIndex}) var ${textureName}: texture_2d<f32>;`
				);
			}
			if (!existingSamplerName) {
				headerBlocks.push(
					`@group(1) @binding(${samplerBindingIndex}) var ${samplerName}: sampler;`
				);
			}
			headerBlocks.push(
				`const ${slotConst}: u32 = ${slot}u;`,
				`const ${uvConst}: u32 = ${uvSet}u;`,
				`const ${linearConst}: bool = ${linear ? "true" : "false"};`
			);
			return {
				header: headerBlocks.join("\n"),
				functions:
					`fn ${fnName}(\n` +
					`\tuv0: vec2<f32>,\n` +
					`\tuv1: vec2<f32>,\n` +
					`\tuv2: vec2<f32>,\n` +
					`\tuv3: vec2<f32>\n` +
					`) -> vec4<f32> {\n` +
					`\tvar uv = uv0;\n` +
					`\tif (${uvConst} == 1u) { uv = uv1; }\n` +
					`\telse if (${uvConst} == 2u) { uv = uv2; }\n` +
					`\telse if (${uvConst} >= 3u) { uv = uv3; }\n` +
					`\tlet sampled = textureSample(${textureName}, ${samplerName}, uv);\n` +
					`\treturn ${decodeExpression};\n` +
					`}`,
				symbols: [
					fnName,
					slotConst,
					uvConst,
					linearConst,
					textureName,
					samplerName,
				],
				headerAnchor: "afterBindings",
				functionsAnchor: "beforeEntryPoint",
			};
		},
	};
}

function createMaterialUniformBlockInjectionScript(): ShaderInjectionScript {
	return {
		id: "ignis/material/uniform-block",
		description:
			"Inject per-material numeric uniform declarations for custom shader materials.",
		run(args, context) {
			if (
				context.sourceKind !== "custom-material" ||
				(context.stage !== "vertex" && context.stage !== "fragment")
			) {
				return null;
			}

			const fields = parseMaterialUniformFields(args.fields);
			if (fields.length <= 0) {
				return null;
			}

			if (context.language === "glsl") {
				const declarations = fields
					.filter((field) => !hasGLSLUniform(context.source, field.webglUniform))
					.map(
						(field) =>
							`uniform ${getGLSLUniformType(field.type)} ${field.webglUniform};`
					);
				if (declarations.length <= 0) {
					return null;
				}
				return {
					header: declarations.join("\n"),
					symbols: fields.map((field) => field.webglUniform),
					headerAnchor: "afterUniforms",
				};
			}

			const structFields = fields
				.map(
					(field) =>
						`\t${field.wgslField}: ${getWGSLUniformType(field.type)},`
				)
				.join("\n");
			const header =
				`struct IgnisShaderUniforms {\n${structFields}\n}\n` +
				`@group(1) @binding(${MATERIAL_SHADER_UNIFORM_BINDING}) ` +
				`var<uniform> ignisShaderUniforms: IgnisShaderUniforms;`;
			return {
				header,
				symbols: [
					"IgnisShaderUniforms",
					"ignisShaderUniforms",
					...fields.map((field) => field.wgslField),
				],
				headerAnchor: "afterStruct",
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
				id: "ignis/webgpu/constants.wgsl",
				code: `#define __WEBGPU_MAX_DIRECTIONAL_LIGHTS__ ${MAX_DIRECTIONAL_LIGHTS}
#define __WEBGPU_MAX_POINT_LIGHTS__ ${MAX_POINT_LIGHTS}
#define __WEBGPU_MAX_SPOT_LIGHTS__ ${MAX_SPOT_LIGHTS}
#define __WEBGPU_MAX_AREA_LIGHTS__ ${MAX_AREA_LIGHTS}
#define __WEBGPU_MAX_LOCAL_LIGHT_PROBES__ ${MAX_LOCAL_LIGHT_PROBES}
#define __WEBGPU_MAX_LOCAL_LIGHT_PROBES__u ${MAX_LOCAL_LIGHT_PROBES}u
#define __WEBGPU_MAX_REFLECTION_PROBES__ ${MAX_REFLECTION_PROBES}
#define __WEBGPU_SH_COEFFICIENT_COUNT__ ${WEBGPU_SH_COEFFICIENT_COUNT}
#define __WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT__ ${WEBGPU_LOCAL_LIGHT_PROBE_COEFFICIENT_COUNT}
#define __WEBGPU_TEXTURE_SLOT_COUNT__ ${WEBGPU_TEXTURE_SLOT_COUNT}
#define __WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT__ ${WEBGPU_FRAME_DIRECTIONAL_LIGHT_VEC4_COUNT}
#define __WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT__ ${WEBGPU_FRAME_POINT_LIGHT_VEC4_COUNT}
#define __WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT__ ${WEBGPU_FRAME_SPOT_LIGHT_VEC4_COUNT}`,
				sourcePath: "runtime://ignis/includes/wgsl/webgpu/constants.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/color/srgb.wgsl",
				code: `fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
	let low = color * vec3<f32>(12.92);
	let high = vec3<f32>(1.055) * pow(color, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
	return select(high, low, color <= vec3<f32>(0.0031308));
}

fn srgbToLinear(color: vec3<f32>) -> vec3<f32> {
	let low = color / vec3<f32>(12.92);
	let high = pow((color + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
	return select(high, low, color <= vec3<f32>(0.04045));
}`,
				sourcePath: "runtime://ignis/includes/wgsl/color/srgb.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/postprocess/fxaa.wgsl",
				code: `const IGNIS_FXAA_EDGE_THRESHOLD_MIN: f32 = ${FXAA_EDGE_THRESHOLD_MIN_LITERAL};
const FXAA_QUALITY = array<f32, ${FXAA_QUALITY.length}>(
	${FXAA_QUALITY_WGSL_LITERAL}
);`,
				sourcePath: "runtime://ignis/includes/wgsl/postprocess/fxaa.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/postprocess/volumetric.wgsl",
				code: `const IGNIS_VOLUMETRIC_SIGMA_T_SCALE: f32 = ${VOLUMETRIC_SIGMA_T_SCALE_LITERAL};`,
				sourcePath: "runtime://ignis/includes/wgsl/postprocess/volumetric.wgsl",
			},
			{
				language: "wgsl",
				id: "ignis/postprocess/fog.wgsl",
				code: `const IGNIS_FOG_MODE_LINEAR: i32 = 0;
const IGNIS_FOG_MODE_EXP: i32 = 1;
const IGNIS_FOG_MODE_EXP2: i32 = 2;

fn ignisFogLinear(depth: f32, startDepth: f32, endDepth: f32) -> f32 {
	let safeRange = max(endDepth - startDepth, 1e-4);
	return clamp((depth - startDepth) / safeRange, 0.0, 1.0);
}

fn ignisFogExp(depth: f32, density: f32) -> f32 {
	let d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-d);
}

fn ignisFogExp2(depth: f32, density: f32) -> f32 {
	let d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-(d * d));
}

fn ignisComputeFogFactor(
	mode: i32,
	depth: f32,
	startDepth: f32,
	endDepth: f32,
	density: f32,
	strength: f32
) -> f32 {
	var fog = ignisFogLinear(depth, startDepth, endDepth);
	if (mode == IGNIS_FOG_MODE_EXP) {
		fog = ignisFogExp(depth, density);
	} else if (mode == IGNIS_FOG_MODE_EXP2) {
		fog = ignisFogExp2(depth, density);
	}
	return clamp(fog * max(strength, 0.0), 0.0, 1.0);
}`,
				sourcePath: "runtime://ignis/includes/wgsl/postprocess/fog.wgsl",
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
		injectionScripts: [
			createPostProcessLumaInjectionScript(),
			createMaterialUniformBlockInjectionScript(),
			createMaterialTextureBindingInjectionScript(),
		],
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
				id: "ignis/webgl/constants.glsl",
				code: `#define __WEBGL_MAX_DIRECTIONAL_LIGHTS__ ${MAX_DIRECTIONAL_LIGHTS}
#define __WEBGL_MAX_POINT_LIGHTS__ ${MAX_POINT_LIGHTS}
#define __WEBGL_MAX_SPOT_LIGHTS__ ${MAX_SPOT_LIGHTS}
#define __WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__ ${MAX_CLUSTER_LIGHTS_PER_FRAGMENT}
#define __WEBGL_MAX_LOCAL_LIGHT_PROBES__ ${MAX_LOCAL_LIGHT_PROBES}
#define __WEBGL_MAX_REFLECTION_PROBES__ ${MAX_REFLECTION_PROBES}`,
				sourcePath: "runtime://ignis/includes/glsl/webgl/constants.glsl",
			},
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
				id: "ignis/postprocess/fxaa.glsl",
				code: `const float IGNIS_FXAA_EDGE_THRESHOLD_MIN = ${FXAA_EDGE_THRESHOLD_MIN_LITERAL};
const float IGNIS_FXAA_QUALITY[${FXAA_QUALITY.length}] = float[${FXAA_QUALITY.length}](${FXAA_QUALITY_GLSL_LITERAL});`,
				sourcePath: "runtime://ignis/includes/glsl/postprocess/fxaa.glsl",
			},
			{
				language: "glsl",
				id: "ignis/postprocess/fog.glsl",
				code: `const int IGNIS_FOG_MODE_LINEAR = 0;
const int IGNIS_FOG_MODE_EXP = 1;
const int IGNIS_FOG_MODE_EXP2 = 2;

float ignisFogLinear(float depth, float startDepth, float endDepth) {
	float safeRange = max(endDepth - startDepth, 1e-4);
	return clamp((depth - startDepth) / safeRange, 0.0, 1.0);
}

float ignisFogExp(float depth, float density) {
	float d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-d);
}

float ignisFogExp2(float depth, float density) {
	float d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-(d * d));
}

float ignisComputeFogFactor(
	int mode,
	float depth,
	float startDepth,
	float endDepth,
	float density,
	float strength
) {
	float fog = ignisFogLinear(depth, startDepth, endDepth);
	if (mode == IGNIS_FOG_MODE_EXP) {
		fog = ignisFogExp(depth, density);
	} else if (mode == IGNIS_FOG_MODE_EXP2) {
		fog = ignisFogExp2(depth, density);
	}
	return clamp(fog * max(strength, 0.0), 0.0, 1.0);
}`,
				sourcePath: "runtime://ignis/includes/glsl/postprocess/fog.glsl",
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
		injectionScripts: [
			createPostProcessLumaInjectionScript(),
			createMaterialUniformBlockInjectionScript(),
			createMaterialTextureBindingInjectionScript(),
		],
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
