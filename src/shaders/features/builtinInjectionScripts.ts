import { defineShaderInjectionScript } from "../runtime/DirectiveProfile";
import type {
	ShaderBackendId,
	ShaderDirectiveFeaturePack,
	ShaderInjectionScript,
	ShaderLanguage,
} from "../runtime/types";
import {
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../backends/webgpu/constants";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_QUALITY,
	VOLUMETRIC_SIGMA_T_SCALE,
} from "../../backends/constants";

const MATERIAL_TEXTURE_SLOT_COUNT = WEBGPU_TEXTURE_SLOT_COUNT;
const MATERIAL_SHADER_UNIFORM_BINDING = WEBGPU_MODEL_BINDING_SHADER_UNIFORMS;
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

function normalizeIdentifierToken(
	value: string | undefined,
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
	value: string | undefined
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

function validateMaterialUniformFields(
	value: string | undefined,
): string | null {
	if (typeof value !== "string" || value.trim().length <= 0) {
		return "argument \"fields\" must contain at least one field.";
	}
	for (const rawField of value.split(";")) {
		const parts = rawField.trim().split(":");
		if (parts.length !== 3) {
			return `field "${rawField}" must use wgslField:type:webglUniform.`;
		}
		const [wgslField, type, webglUniform] = parts;
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(wgslField)) {
			return `WGSL field "${wgslField}" must be a shader identifier.`;
		}
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(webglUniform)) {
			return `WebGL uniform "${webglUniform}" must be a shader identifier.`;
		}
		if (normalizeMaterialUniformType(type) !== type) {
			return `uniform field "${wgslField}" has unsupported type "${type}".`;
		}
	}
	return null;
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
	return defineShaderInjectionScript({
		id: "ignis/postprocess/luma",
		description:
			"Inject a shared luma() implementation with configurable profile.",
		arguments: {
			profile: {
				type: "enum",
				values: ["bt601", "bt709"],
				default: "bt709",
			},
			clamp: { type: "boolean", default: true },
		},
		run(args, context) {
			const profile = (args.profile ?? "bt709") as LumaProfile;
			const clampInput = args.clamp ?? true;
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
	});
}

function createPostProcessFXAAInjectionScript(): ShaderInjectionScript {
	return defineShaderInjectionScript({
		id: "ignis/postprocess/fxaa",
		description: "Inject FXAA constants into the owning post-process shader.",
		arguments: {},
		run(_args, context) {
			const threshold = toShaderFloat(FXAA_EDGE_THRESHOLD_MIN);
			if (context.language === "glsl") {
				return {
					header: `const float IGNIS_FXAA_EDGE_THRESHOLD_MIN = ${threshold};`,
					headerAnchor: "afterPrecision",
				};
			}
			const quality = FXAA_QUALITY.map(toShaderFloat).join(", ");
			return {
				header:
					`const IGNIS_FXAA_EDGE_THRESHOLD_MIN: f32 = ${threshold};\n` +
					`const FXAA_QUALITY = array<f32, ${FXAA_QUALITY.length}>(\n` +
					`\t${quality}\n` +
					");",
				headerAnchor: "afterEnable",
			};
		},
	});
}

function createPostProcessVolumetricInjectionScript(): ShaderInjectionScript {
	return defineShaderInjectionScript({
		id: "ignis/postprocess/volumetric",
		language: "wgsl",
		description: "Inject volumetric constants into the owning post-process shader.",
		arguments: {},
		run() {
			return {
				header:
					"const IGNIS_VOLUMETRIC_SIGMA_T_SCALE: f32 = " +
					`${toShaderFloat(VOLUMETRIC_SIGMA_T_SCALE)};`,
				headerAnchor: "afterEnable",
			};
		},
	});
}

function createMaterialTextureBindingInjectionScript(): ShaderInjectionScript {
	return defineShaderInjectionScript({
		id: "ignis/material/texture-binding",
		description:
			"Inject per-material texture helper declarations for custom shader materials.",
		arguments: {
			name: { type: "string", required: true },
			slot: {
				type: "integer",
				required: true,
				min: 0,
				max: MATERIAL_TEXTURE_SLOT_COUNT - 1,
			},
			uv: { type: "integer", default: 0, min: 0, max: 3 },
			linear: { type: "boolean", default: false },
			uniform: { type: "string", required: true },
		},
		validateArguments(args) {
			return typeof args.uniform === "string" &&
				/^[A-Za-z_][A-Za-z0-9_]*$/.test(args.uniform) ?
					null
				:	"argument \"uniform\" must be a shader identifier.";
		},
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
			const slot = args.slot ?? 0;
			const uvSet = args.uv ?? 0;
			const linear = args.linear ?? false;
			const symbolToken = nameToken.toUpperCase();
			const fnName = `ignisSampleTexture_${nameToken}`;
			const fnLevelName = `ignisSampleTextureLevel_${nameToken}`;
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
						`}\n\n` +
						`vec4 ${fnLevelName}(vec2 uv0, vec2 uv1, vec2 uv2, vec2 uv3, float lod) {\n` +
						`\tvec2 uv = uv0;\n` +
						`\tif (${uvConst} == 1) uv = uv1;\n` +
						`\telse if (${uvConst} == 2) uv = uv2;\n` +
						`\telse if (${uvConst} >= 3) uv = uv3;\n` +
						`\tvec4 sampled = textureLod(${uniformName}, uv, lod);\n` +
						`\treturn ${decodeExpression};\n` +
						`}`,
					symbols: [fnName, fnLevelName, slotConst, uvConst, linearConst, uniformName],
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
					`}\n\n` +
					`fn ${fnLevelName}(\n` +
					`\tuv0: vec2<f32>,\n` +
					`\tuv1: vec2<f32>,\n` +
					`\tuv2: vec2<f32>,\n` +
					`\tuv3: vec2<f32>,\n` +
					`\tlevel: f32\n` +
					`) -> vec4<f32> {\n` +
					`\tvar uv = uv0;\n` +
					`\tif (${uvConst} == 1u) { uv = uv1; }\n` +
					`\telse if (${uvConst} == 2u) { uv = uv2; }\n` +
					`\telse if (${uvConst} >= 3u) { uv = uv3; }\n` +
					`\tlet sampled = textureSampleLevel(${textureName}, ${samplerName}, uv, level);\n` +
					`\treturn ${decodeExpression};\n` +
					`}`,
				symbols: [
					fnName,
					fnLevelName,
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
	});
}

function createMaterialUniformBlockInjectionScript(): ShaderInjectionScript {
	return defineShaderInjectionScript({
		id: "ignis/material/uniform-block",
		description:
			"Inject per-material numeric uniform declarations for custom shader materials.",
		arguments: {
			fields: { type: "string", required: true },
		},
		validateArguments(args) {
			return validateMaterialUniformFields(args.fields);
		},
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
	});
}

export function createBuiltinInjectionFeaturePacks(
	backend: ShaderBackendId,
): ShaderDirectiveFeaturePack[] {
	return [
		{
			id: "ignis/postprocess-injections",
			backend,
			revision: backend === "webgpu" ? 3 : 2,
			includeModules: [],
			injectionScripts: [
				createPostProcessLumaInjectionScript(),
				createPostProcessFXAAInjectionScript(),
				...(backend === "webgpu" ?
					[createPostProcessVolumetricInjectionScript()]
				:	[]),
			],
		},
		{
			id: "ignis/material-injections",
			backend,
			revision: 1,
			includeModules: [],
			injectionScripts: [
				createMaterialUniformBlockInjectionScript(),
				createMaterialTextureBindingInjectionScript(),
			],
		},
	];
}
