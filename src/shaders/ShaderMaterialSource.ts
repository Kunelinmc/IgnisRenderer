import type {
	ResolvedShaderMaterialTextureBinding,
	ResolvedShaderMaterialUniformBinding,
	ShaderMaterial,
	ShaderMaterialUniformType,
} from "../materials/ShaderMaterial";
import { ShaderSource } from "./ShaderSource";
import type {
	ShaderGeneratedSourceBlock,
	ShaderLanguage,
} from "./runtime";

export const SHADER_MATERIAL_SOURCE_ABI_REVISION = 1;

export interface ShaderMaterialSourceOptions {
	readonly material: ShaderMaterial;
	readonly language: ShaderLanguage;
	readonly stage: "vertex" | "fragment";
	readonly source: string;
	readonly wgslUniformBinding?: number;
}

function normalizeIdentifierToken(value: string): string {
	const sanitized = value.replace(/[^A-Za-z0-9_]/g, "_");
	return /^[A-Za-z_]/.test(sanitized) ? sanitized : `x_${sanitized}`;
}

function assertTextureNameTokenAvailable(
	usedTokens: Set<string>,
	name: string,
): string {
	const token = normalizeIdentifierToken(name);
	if (usedTokens.has(token)) {
		throw new Error(
			`ShaderMaterial texture names collide on generated token "${token}".`,
		);
	}
	usedTokens.add(token);
	return token;
}

function getWGSLUniformType(type: ShaderMaterialUniformType): string {
	switch (type) {
		case "i32": return "i32";
		case "u32": return "u32";
		case "vec2f": return "vec2<f32>";
		case "vec3f": return "vec3<f32>";
		case "vec4f": return "vec4<f32>";
		case "vec2i": return "vec2<i32>";
		case "vec3i": return "vec3<i32>";
		case "vec4i": return "vec4<i32>";
		case "vec2u": return "vec2<u32>";
		case "vec3u": return "vec3<u32>";
		case "vec4u": return "vec4<u32>";
		case "mat4x4f": return "mat4x4<f32>";
		case "f32":
		default:
			return "f32";
	}
}

function getGLSLUniformType(type: ShaderMaterialUniformType): string {
	switch (type) {
		case "i32": return "int";
		case "u32": return "uint";
		case "vec2f": return "vec2";
		case "vec3f": return "vec3";
		case "vec4f": return "vec4";
		case "vec2i": return "ivec2";
		case "vec3i": return "ivec3";
		case "vec4i": return "ivec4";
		case "vec2u": return "uvec2";
		case "vec3u": return "uvec3";
		case "vec4u": return "uvec4";
		case "mat4x4f": return "mat4";
		case "f32":
		default:
			return "float";
	}
}

function createPaddingField(
	stage: "vertex" | "fragment",
	index: number,
	usedFields: Set<string>,
): string {
	let suffix = 0;
	let field = `__ignisPad_${stage}_${index}`;
	while (usedFields.has(field)) {
		suffix++;
		field = `__ignisPad_${stage}_${index}_${suffix}`;
	}
	usedFields.add(field);
	return field;
}

function resolveWGSLUniformLayout(
	bindings: ResolvedShaderMaterialUniformBinding[],
	stage: "vertex" | "fragment",
): ResolvedShaderMaterialUniformBinding[] {
	const usedFields = new Set(bindings.map((binding) => binding.wgslField));
	return bindings.map((binding, index) =>
		binding.stage === "both" || binding.stage === stage ?
			binding
			: {
				...binding,
				wgslField: createPaddingField(stage, index, usedFields),
			}
	);
}

function assertWGSLBindingAvailable(source: string, binding: number): void {
	const attributes =
		"(?:@[A-Za-z_][A-Za-z0-9_]*(?:\\s*\\([^)]*\\))?\\s*)*";
	const patterns = [
		new RegExp(
			`@group\\s*\\(\\s*1\\s*\\)\\s*` +
				`@binding\\s*\\(\\s*${binding}\\s*\\)\\s*` +
				`${attributes}var(?:<[^>]+>)?\\s+[A-Za-z_][A-Za-z0-9_]*\\s*:`,
			"m",
		),
		new RegExp(
			`@binding\\s*\\(\\s*${binding}\\s*\\)\\s*` +
				"@group\\s*\\(\\s*1\\s*\\)\\s*" +
				`${attributes}var(?:<[^>]+>)?\\s+[A-Za-z_][A-Za-z0-9_]*\\s*:`,
			"m",
		),
	];
	if (patterns.some((pattern) => pattern.test(source))) {
		throw new Error(
			`ShaderMaterial WGSL source redeclares reserved @group(1) @binding(${binding}).`,
		);
	}
}

function assertWGSLDeclarationAvailable(
	source: string,
	declaration: RegExp,
	name: string,
): void {
	if (declaration.test(source)) {
		throw new Error(
			`ShaderMaterial WGSL source redeclares reserved symbol "${name}".`,
		);
	}
}

function createWGSLUniformBlocks(
	bindings: ResolvedShaderMaterialUniformBinding[],
	stage: "vertex" | "fragment",
	source: string,
	uniformBinding: number,
): ShaderGeneratedSourceBlock[] {
	if (bindings.length <= 0) return [];
	assertWGSLBindingAvailable(source, uniformBinding);
	assertWGSLDeclarationAvailable(
		source,
		/\bstruct\s+IgnisShaderUniforms\b/m,
		"IgnisShaderUniforms",
	);
	assertWGSLDeclarationAvailable(
		source,
		/\bvar(?:<[^>]+>)?\s+ignisShaderUniforms\b/m,
		"ignisShaderUniforms",
	);
	const fields = resolveWGSLUniformLayout(bindings, stage)
		.map((binding) =>
			`\t${binding.wgslField}: ${getWGSLUniformType(binding.type)},`
		)
		.join("\n");
	return [{
		code:
			`struct IgnisShaderUniforms {\n${fields}\n}\n` +
			`@group(1) @binding(${uniformBinding}) ` +
			"var<uniform> ignisShaderUniforms: IgnisShaderUniforms;",
		sourcePath: "<generated:webgpu:shader-material:uniforms>",
		label: "webgpu-shader-material-uniforms",
		anchor: "afterStruct",
	}];
}

function createWGSLTextureBlocks(
	bindings: ResolvedShaderMaterialTextureBinding[],
	source: string,
): ShaderGeneratedSourceBlock[] {
	if (bindings.length <= 0) return [];
	for (const [name, declaration] of [
		["ignisSelectShaderMaterialUv", /\bfn\s+ignisSelectShaderMaterialUv\b/m],
		["ignisDecodeShaderMaterialSample", /\bfn\s+ignisDecodeShaderMaterialSample\b/m],
	] as const) {
		assertWGSLDeclarationAvailable(source, declaration, name);
	}
	const declarations: string[] = [];
	const wrappers: string[] = [];
	const usedTokens = new Set<string>();
	for (const binding of bindings) {
		const token = assertTextureNameTokenAvailable(usedTokens, binding.name);
		const symbolToken = token.toUpperCase();
		const textureName = `ignisShaderTexture_${token}`;
		const samplerName = `ignisShaderSampler_${token}`;
		const functionName = `ignisSampleTexture_${token}`;
		const levelFunctionName = `ignisSampleTextureLevel_${token}`;
		const slotConst = `IGNIS_TEXTURE_SLOT_${symbolToken}`;
		const uvConst = `IGNIS_TEXTURE_UVSET_${symbolToken}`;
		const linearConst = `IGNIS_TEXTURE_LINEAR_${symbolToken}`;
		const textureBinding = binding.slot * 2 + 1;
		const samplerBinding = textureBinding + 1;
		assertWGSLBindingAvailable(source, textureBinding);
		assertWGSLBindingAvailable(source, samplerBinding);
		for (const [name, declaration] of [
			[textureName, new RegExp(`\\bvar(?:<[^>]+>)?\\s+${textureName}\\b`, "m")],
			[samplerName, new RegExp(`\\bvar(?:<[^>]+>)?\\s+${samplerName}\\b`, "m")],
			[functionName, new RegExp(`\\bfn\\s+${functionName}\\b`, "m")],
			[levelFunctionName, new RegExp(`\\bfn\\s+${levelFunctionName}\\b`, "m")],
		] as const) {
			assertWGSLDeclarationAvailable(source, declaration, name);
		}
		declarations.push(
			`@group(1) @binding(${textureBinding}) var ${textureName}: texture_2d<f32>;`,
			`@group(1) @binding(${samplerBinding}) var ${samplerName}: sampler;`,
			`const ${slotConst}: u32 = ${binding.slot}u;`,
			`const ${uvConst}: u32 = ${binding.uvSet}u;`,
			`const ${linearConst}: bool = ${binding.linear ? "true" : "false"};`,
		);
		wrappers.push(
			`fn ${functionName}(\n` +
			"\tuv0: vec2<f32>,\n\tuv1: vec2<f32>,\n" +
			"\tuv2: vec2<f32>,\n\tuv3: vec2<f32>\n" +
			") -> vec4<f32> {\n" +
			`\tlet uv = ignisSelectShaderMaterialUv(` +
			`uv0, uv1, uv2, uv3, ${uvConst});\n` +
			`\tlet sampled = textureSample(${textureName}, ${samplerName}, uv);\n` +
			`\treturn ignisDecodeShaderMaterialSample(sampled, ${linearConst});\n}\n\n` +
			`fn ${levelFunctionName}(\n` +
			"\tuv0: vec2<f32>,\n\tuv1: vec2<f32>,\n" +
			"\tuv2: vec2<f32>,\n\tuv3: vec2<f32>,\n\tlevel: f32\n" +
			") -> vec4<f32> {\n" +
			`\tlet uv = ignisSelectShaderMaterialUv(` +
			`uv0, uv1, uv2, uv3, ${uvConst});\n` +
			`\tlet sampled = textureSampleLevel(` +
			`${textureName}, ${samplerName}, uv, level);\n` +
			`\treturn ignisDecodeShaderMaterialSample(sampled, ${linearConst});\n}`,
		);
	}
	return [
		{
			code: declarations.join("\n"),
			sourcePath: "<generated:webgpu:shader-material:textures>",
			label: "webgpu-shader-material-textures",
			anchor: "afterBindings",
		},
		{
			code: ShaderSource.getSync("webgpu.material.textureHelpers").source.code,
			sourcePath: "./webgpu/material/shaderMaterialTextureHelpers.wgsl",
			label: "webgpu-shader-material-shared-texture-helpers",
			anchor: "beforeEntryPoint",
		},
		{
			code: wrappers.join("\n\n"),
			sourcePath: "<generated:webgpu:shader-material:texture-helpers>",
			label: "webgpu-shader-material-texture-helpers",
			anchor: "beforeEntryPoint",
		},
	];
}

function assertGLSLUniformAvailable(source: string, uniformName: string): void {
	const declaration = new RegExp(
		"\\buniform\\s+(?:(?:lowp|mediump|highp)\\s+)?" +
			`[A-Za-z_][A-Za-z0-9_]*\\s+${uniformName}\\b`,
		"m",
	);
	if (declaration.test(source)) {
		throw new Error(
			`ShaderMaterial GLSL source redeclares generated uniform "${uniformName}".`,
		);
	}
}

function createGLSLUniformBlocks(
	bindings: ResolvedShaderMaterialUniformBinding[],
	stage: "vertex" | "fragment",
	source: string,
): ShaderGeneratedSourceBlock[] {
	const active = bindings.filter(
		(binding) => binding.stage === "both" || binding.stage === stage,
	);
	if (active.length <= 0) return [];
	for (const binding of active) {
		assertGLSLUniformAvailable(source, binding.webglUniform);
	}
	return [{
		code: active
			.map((binding) =>
				`uniform ${getGLSLUniformType(binding.type)} ${binding.webglUniform};`
			)
			.join("\n"),
		sourcePath: "<generated:webgl:shader-material:uniforms>",
		label: "webgl-shader-material-uniforms",
		anchor: "afterUniforms",
	}];
}

function createGLSLTextureBlocks(
	bindings: ResolvedShaderMaterialTextureBinding[],
	source: string,
): ShaderGeneratedSourceBlock[] {
	if (bindings.length <= 0) return [];
	for (const name of [
		"ignisSelectShaderMaterialUv",
		"ignisDecodeShaderMaterialSample",
	]) {
		if (new RegExp(`\\b(?:vec[234]\\s+)?${name}\\s*\\(`, "m").test(source)) {
			throw new Error(
				`ShaderMaterial GLSL source redeclares reserved helper "${name}".`,
			);
		}
	}
	const declarations: string[] = [];
	const wrappers: string[] = [];
	const usedTokens = new Set<string>();
	for (const binding of bindings) {
		assertGLSLUniformAvailable(source, binding.webglUniform);
		const token = assertTextureNameTokenAvailable(usedTokens, binding.name);
		const symbolToken = token.toUpperCase();
		const functionName = `ignisSampleTexture_${token}`;
		const levelFunctionName = `ignisSampleTextureLevel_${token}`;
		for (const name of [functionName, levelFunctionName]) {
			if (new RegExp(`\\b(?:vec4\\s+)?${name}\\s*\\(`, "m").test(source)) {
				throw new Error(
					`ShaderMaterial GLSL source redeclares reserved helper "${name}".`,
				);
			}
		}
		const slotConst = `IGNIS_TEXTURE_SLOT_${symbolToken}`;
		const uvConst = `IGNIS_TEXTURE_UVSET_${symbolToken}`;
		const linearConst = `IGNIS_TEXTURE_LINEAR_${symbolToken}`;
		declarations.push(
			`uniform sampler2D ${binding.webglUniform};`,
			`const int ${slotConst} = ${binding.slot};`,
			`const int ${uvConst} = ${binding.uvSet};`,
			`const bool ${linearConst} = ${binding.linear ? "true" : "false"};`,
		);
		wrappers.push(
			`vec4 ${functionName}(vec2 uv0, vec2 uv1, vec2 uv2, vec2 uv3) {\n` +
			`\tvec2 uv = ignisSelectShaderMaterialUv(` +
			`uv0, uv1, uv2, uv3, ${uvConst});\n` +
			`\tvec4 sampled = texture(${binding.webglUniform}, uv);\n` +
			`\treturn ignisDecodeShaderMaterialSample(sampled, ${linearConst});\n}\n\n` +
			`vec4 ${levelFunctionName}(` +
			"vec2 uv0, vec2 uv1, vec2 uv2, vec2 uv3, float lod) {\n" +
			`\tvec2 uv = ignisSelectShaderMaterialUv(` +
			`uv0, uv1, uv2, uv3, ${uvConst});\n` +
			`\tvec4 sampled = textureLod(${binding.webglUniform}, uv, lod);\n` +
			`\treturn ignisDecodeShaderMaterialSample(sampled, ${linearConst});\n}`,
		);
	}
	return [
		{
			code: declarations.join("\n"),
			sourcePath: "<generated:webgl:shader-material:textures>",
			label: "webgl-shader-material-textures",
			anchor: "afterUniforms",
		},
		{
			code: ShaderSource.getSync("webgl.material.textureHelpers").source.code,
			sourcePath: "./webgl/material/shaderMaterialTextureHelpers.glsl",
			label: "webgl-shader-material-shared-texture-helpers",
			anchor: "beforeEntryPoint",
		},
		{
			code: wrappers.join("\n\n"),
			sourcePath: "<generated:webgl:shader-material:texture-helpers>",
			label: "webgl-shader-material-texture-helpers",
			anchor: "beforeEntryPoint",
		},
	];
}

/** @internal Compiles structured material bindings into backend shader blocks. */
export function createShaderMaterialSourceBlocks(
	options: ShaderMaterialSourceOptions,
): ShaderGeneratedSourceBlock[] {
	const uniforms = options.material.getUniformBindings();
	const textures = options.material.getTextureBindings();
	if (options.language === "wgsl") {
		if (!Number.isInteger(options.wgslUniformBinding)) {
			throw new Error("WGSL ShaderMaterial source requires a uniform binding.");
		}
		const blocks = createWGSLUniformBlocks(
			uniforms,
			options.stage,
			options.source,
			options.wgslUniformBinding as number,
		);
		if (options.stage === "fragment") {
			blocks.push(...createWGSLTextureBlocks(textures, options.source));
		}
		return blocks;
	}
	const blocks = createGLSLUniformBlocks(
		uniforms,
		options.stage,
		options.source,
	);
	if (options.stage === "fragment") {
		blocks.push(...createGLSLTextureBlocks(textures, options.source));
	}
	return blocks;
}
