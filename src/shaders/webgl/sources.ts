import {
	createShaderModuleGroup,
	getShaderManifestSourceKeys,
	mapShaderManifestEntries,
	type ShaderBackendManifest,
	type ShaderManifestExpression,
	type ShaderParameterSchema,
	type ShaderParameterValue,
	type ShaderSourceNode,
} from "../ShaderManifest";
import {
	WEBGL_ALPHA_MAP_DEPTH_VARIANT,
	WEBGL_FULL_SCENE_VARIANT,
} from "./sceneVariants";

export type WebGLShaderPart = keyof typeof WEBGL_SHADER_FILES;
export type WebGLSceneFragmentPart = keyof typeof WEBGL_SCENE_FRAGMENT_SHADER_FILES;
export type WebGLDirectiveShaderPart = keyof typeof WEBGL_DIRECTIVE_SHADER_FILES;
export type WebGLDeformationSpecialization = ShaderParameterValue<typeof deformationSchema>;

const WEBGL_SHADER_FILES = {
	sceneVertex: "./webgl/scene/sceneVertex.glsl",
	sceneDepthPrepassVertex: "./webgl/scene/sceneDepthPrepassVertex.glsl",
	sceneDepthPrepassFragment: "./webgl/scene/sceneDepthPrepassFragment.glsl",
	environmentVertex: "./webgl/environment/environmentVertex.glsl",
	environmentFragment: "./webgl/environment/environmentFragment.glsl",
	iblPrefilterFragment: "./webgl/environment/iblPrefilterFragment.glsl",
	presentVertex: "./webgl/utility/presentVertex.glsl",
	presentFragment: "./webgl/utility/presentFragment.glsl",
	particleVertex: "./webgl/particles/particleVertex.glsl",
	particleFragment: "./webgl/particles/particleFragment.glsl",
	shadowDepthVertex: "./webgl/shadow/shadowDepthVertex.glsl",
	shadowDepthFragment: "./webgl/shadow/shadowDepthFragment.glsl",
	shadowTransmittanceFragment: "./webgl/shadow/shadowTransmittanceFragment.glsl",
	copyFragment: "./webgl/postprocess/copyFragment.glsl",
	oitResolveFragment: "./webgl/utility/oitResolveFragment.glsl",
	gammaFragment: "./webgl/postprocess/gammaFragment.glsl",
	toneMappingFragment: "./webgl/postprocess/toneMappingFragment.glsl",
	colorFilterFragment: "./webgl/postprocess/colorFilterFragment.glsl",
	fxaaFragment: "./webgl/postprocess/fxaaFragment.glsl",
	bloomFragment: "./webgl/postprocess/bloomFragment.glsl",
	motionBlurFragment: "./webgl/postprocess/motionBlurFragment.glsl",
	fogFragment: "./webgl/postprocess/fogFragment.glsl",
	dofFragment: "./webgl/postprocess/dofFragment.glsl",
	taaFragment: "./webgl/postprocess/taaFragment.glsl",
	ssaoRawFragment: "./webgl/postprocess/ssaoRawFragment.glsl",
	ssaoBlurFragment: "./webgl/postprocess/ssaoBlurFragment.glsl",
	ssaoCombineFragment: "./webgl/postprocess/ssaoCombineFragment.glsl",
} as const satisfies Record<string, string>;

const WEBGL_SCENE_FRAGMENT_SHADER_FILES = {
	fragmentPrelude: "./webgl/scene/fragmentPrelude.glsl",
	fragmentUniforms: "./webgl/scene/fragmentUniforms.glsl",
	fragmentMaterialBlocks: "./webgl/scene/fragmentMaterialBlocks.glsl",
	fragmentUvTextureNormal: "./webgl/scene/fragmentUvTextureNormal.glsl",
	fragmentSh: "./webgl/scene/fragmentSh.glsl",
	fragmentLocalProbes: "./webgl/scene/fragmentLocalProbes.glsl",
	fragmentClusteredLighting:
		"./webgl/scene/fragmentClusteredLighting.glsl",
	fragmentReflectionEnvironment:
		"./webgl/scene/fragmentReflectionEnvironment.glsl",
	fragmentEnvironmentSpecular:
		"./webgl/scene/fragmentEnvironmentSpecular.glsl",
	fragmentReflectionProbes: "./webgl/scene/fragmentReflectionProbes.glsl",
	fragmentLightAttenuation: "./webgl/scene/fragmentLightAttenuation.glsl",
	fragmentShadows: "./webgl/scene/fragmentShadows.glsl",
	fragmentBrdfPbr: "./webgl/scene/fragmentBrdfPbr.glsl",
	fragmentPhong: "./webgl/scene/fragmentPhong.glsl",
	fragmentPbrLighting: "./webgl/scene/fragmentPbrLighting.glsl",
	fragmentMainOutput: "./webgl/scene/fragmentMainOutput.glsl",
} as const satisfies Record<string, string>;

export const WEBGL_SCENE_FRAGMENT_PARTS: readonly WebGLSceneFragmentPart[] =
	Object.keys(WEBGL_SCENE_FRAGMENT_SHADER_FILES) as WebGLSceneFragmentPart[];

const WEBGL_INTERNAL_SHADER_FILES = {
	diffuseProbeFallbackFragment: "./webgl/environment/diffuseProbeFallbackFragment.glsl",
	irradianceProbeGridFragment: "./webgl/environment/irradianceProbeGridFragment.glsl",
} as const;

const WEBGL_DIRECTIVE_SHADER_FILES = {
	animation: { path: "./webgl/common/animation.glsl", sync: true },
	constants: "./webgl/directives/constants.glsl",
	srgb: "./webgl/directives/srgb.glsl",
	lumaWeights: "./webgl/directives/lumaWeights.glsl",
	lumaCommon: "./webgl/directives/lumaCommon.glsl",
} as const;

const literal = (value: string | number | boolean): ShaderManifestExpression => ({ literal: value });
const parameter = (path: string): ShaderManifestExpression => ({ parameter: path });
const equals = (path: string, value: string): ShaderManifestExpression => ({
	equals: [parameter(path), literal(value)],
});
const all = (...values: ShaderManifestExpression[]): ShaderManifestExpression => ({ all: values });
const any = (...values: ShaderManifestExpression[]): ShaderManifestExpression => ({ any: values });
const not = (value: ShaderManifestExpression): ShaderManifestExpression => ({ not: value });
const asset = <const T extends string>(id: T) => ({ asset: id });
const when = (
	condition: ShaderManifestExpression,
	then: ShaderSourceNode,
	otherwise?: ShaderSourceNode,
): ShaderSourceNode => ({ when: condition, then, else: otherwise });

const scene = (path: string): ShaderManifestExpression =>
	parameter(`specialization.scene.${path}`);
const material = (path: string): ShaderManifestExpression =>
	parameter(`specialization.material.${path}`);
const model = (value: string): ShaderManifestExpression =>
	equals("specialization.material.model", value);
const isFull = model("full");
const isPBR = any(model("pbr"), isFull);
const isPhong = model("phong");
const isFlat = model("flat");
const isLegacy = any(isPhong, isFlat, isFull);
const isLit = any(isPBR, isLegacy);
const shadowTransmittance = all(scene("shadows"), scene("shadowTransmittance"));
const irradianceGrid = all(
	scene("sh"),
	scene("localLightProbes"),
	scene("irradianceProbeGrid"),
);

const { model: _materialModel, ...materialFeatures } = WEBGL_FULL_SCENE_VARIANT.material;
const materialFields = mapShaderManifestEntries(materialFeatures, "",
	() => ({ type: "boolean", default: false } as const));

const sceneSpecializationSchema = {
	type: "record",
	default: WEBGL_FULL_SCENE_VARIANT as unknown as Record<string, unknown>,
	fields: {
		output: { type: "enum", values: ["single", "mrt"], default: "mrt" },
		materialGBuffer: { type: "boolean", default: false },
		oit: { type: "boolean", default: false },
		scene: {
			type: "record",
			fields: mapShaderManifestEntries(WEBGL_FULL_SCENE_VARIANT.scene, "",
				() => ({ type: "boolean", default: false } as const)),
		},
		material: {
			type: "record",
			fields: {
				model: {
					type: "enum",
					values: ["unlit", "flat", "phong", "pbr", "full"],
					default: "pbr",
				},
				...materialFields,
			},
		},
		skinProfile: {
			type: "enum",
			values: ["static", "skin4", "skin8"],
			default: "static",
		},
		morphSemanticMask: { type: "integer", default: 0, min: 0, bitMask: 3 },
	},
} as const satisfies ShaderParameterSchema;

const sceneParams = {
	type: "record",
	fields: { specialization: sceneSpecializationSchema },
} as const satisfies ShaderParameterSchema;

const deformationSchema = {
	type: "record",
	fields: {
		skinProfile: {
			type: "enum",
			values: ["static", "skin4", "skin8"],
			default: "static",
		},
		morphPosition: { type: "boolean", default: false },
	},
} as const satisfies ShaderParameterSchema;

const depthParams = {
	type: "record",
	fields: {
		specialization: {
			type: "record",
			default: WEBGL_ALPHA_MAP_DEPTH_VARIANT as unknown as Record<string, unknown>,
			fields: {
				alphaMask: { type: "boolean", default: false },
				baseMap: { type: "boolean", default: false },
				...deformationSchema.fields,
			},
		},
	},
} as const satisfies ShaderParameterSchema;

const shadowParams = {
	type: "record",
	fields: { specialization: deformationSchema },
} as const satisfies ShaderParameterSchema;

const skinInfluences = (prefix: string): ShaderManifestExpression => ({
	select: {
		cases: [
			{ when: equals(`${prefix}.skinProfile`, "skin8"), value: literal(8) },
			{ when: equals(`${prefix}.skinProfile`, "skin4"), value: literal(4) },
		],
		fallback: literal(0),
	},
});

const animationDefines = (
	prefix: string,
	hasMorph: ShaderManifestExpression,
	path: string,
): ShaderSourceNode => ({
	defines: {
		IGNIS_WEBGL_DEFORMATION_ACTIVE: any(
			not(equals(`${prefix}.skinProfile`, "static")),
			hasMorph,
		),
		IGNIS_WEBGL_SKIN_INFLUENCES: skinInfluences(prefix),
	},
	sourcePath: path,
});

const sceneDefines: ShaderSourceNode = {
	defines: {
		WEBGL_SCENE_OUTPUT_MRT: equals("specialization.output", "mrt"),
		WEBGL_SCENE_OUTPUT_MATERIAL_GBUFFER: parameter("specialization.materialGBuffer"),
		WEBGL_SCENE_OIT: parameter("specialization.oit"),
		WEBGL_SCENE_SHADOWS: scene("shadows"),
		WEBGL_SCENE_SHADOW_TRANSMITTANCE: scene("shadowTransmittance"),
		WEBGL_SCENE_CLUSTERED_LIGHTING: scene("clusteredLighting"),
		WEBGL_SCENE_SH: scene("sh"),
		WEBGL_SCENE_LOCAL_LIGHT_PROBES: scene("localLightProbes"),
		WEBGL_SCENE_IRRADIANCE_PROBE_GRID: scene("irradianceProbeGrid"),
		WEBGL_SCENE_REFLECTION_PROBES: scene("reflectionProbes"),
		WEBGL_SCENE_ENVIRONMENT_SPECULAR: scene("environmentSpecular"),
		WEBGL_MATERIAL_MODEL_UNLIT: model("unlit"),
		WEBGL_MATERIAL_MODEL_FLAT: isFlat,
		WEBGL_MATERIAL_MODEL_PHONG: isPhong,
		WEBGL_MATERIAL_MODEL_LEGACY: any(isPhong, isFlat),
		WEBGL_MATERIAL_MODEL_PBR: model("pbr"),
		WEBGL_MATERIAL_MODEL_FULL: isFull,
		...Object.fromEntries(
			Object.keys(WEBGL_FULL_SCENE_VARIANT.material)
				.filter((name) => name !== "model")
				.map((name) => [
					`WEBGL_MATERIAL_${name.replace(/([A-Z])/g, "_$1").toUpperCase()}`,
					material(name),
				]),
		),
	},
	sourcePath: "<webgl-scene-specialization-defines>",
};

const uniformMarkers: readonly [string, ShaderManifestExpression][] = [
	["__WEBGL_MATERIAL_COMMON_LEGACY_UNIFORMS__", isFull],
	["__WEBGL_SCENE_LIGHTING_UNIFORMS__", isLit],
	["__WEBGL_SCENE_SH_UNIFORMS__", scene("sh")],
	["__WEBGL_MATERIAL_SHADING_MODEL_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_PBR_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_SPECULAR_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_SPECULAR_MAP_UNIFORMS__", all(isPBR, material("specularMap"))],
	["__WEBGL_MATERIAL_SPECULAR_MAP_LEGACY_UNIFORMS__", all(isFull, material("specularMap"))],
	["__WEBGL_MATERIAL_SPECULAR_COLOR_MAP_UNIFORMS__", all(isPBR, material("specularColorMap"))],
	["__WEBGL_MATERIAL_SPECULAR_COLOR_MAP_LEGACY_UNIFORMS__", all(isFull, material("specularColorMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_CLEARCOAT_MAP_UNIFORMS__", all(isPBR, material("clearcoatMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_MAP_LEGACY_UNIFORMS__", all(isFull, material("clearcoatMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_ROUGHNESS_MAP_UNIFORMS__", all(isPBR, material("clearcoatRoughnessMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_ROUGHNESS_MAP_LEGACY_UNIFORMS__", all(isFull, material("clearcoatRoughnessMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_NORMAL_MAP_UNIFORMS__", all(isPBR, material("clearcoatNormalMap"))],
	["__WEBGL_MATERIAL_CLEARCOAT_NORMAL_MAP_LEGACY_UNIFORMS__", all(isFull, material("clearcoatNormalMap"))],
	["__WEBGL_MATERIAL_SHEEN_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_SHEEN_COLOR_MAP_UNIFORMS__", all(isPBR, material("sheenColorMap"))],
	["__WEBGL_MATERIAL_SHEEN_COLOR_MAP_LEGACY_UNIFORMS__", all(isFull, material("sheenColorMap"))],
	["__WEBGL_MATERIAL_SHEEN_ROUGHNESS_MAP_UNIFORMS__", all(isPBR, material("sheenRoughnessMap"))],
	["__WEBGL_MATERIAL_SHEEN_ROUGHNESS_MAP_LEGACY_UNIFORMS__", all(isFull, material("sheenRoughnessMap"))],
	["__WEBGL_MATERIAL_TRANSMISSION_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_TRANSMISSION_RUNTIME_UNIFORMS__", all(isPBR, material("transmission"))],
	["__WEBGL_MATERIAL_TRANSMISSION_MAP_UNIFORMS__", all(isPBR, material("transmissionMap"))],
	["__WEBGL_MATERIAL_TRANSMISSION_MAP_LEGACY_UNIFORMS__", all(isFull, material("transmissionMap"))],
	["__WEBGL_MATERIAL_THICKNESS_MAP_UNIFORMS__", all(isPBR, material("thicknessMap"))],
	["__WEBGL_MATERIAL_THICKNESS_MAP_LEGACY_UNIFORMS__", all(isFull, material("thicknessMap"))],
	["__WEBGL_MATERIAL_IRIDESCENCE_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_ANISOTROPY_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_PHONG_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_ALPHA_UNIFORMS__", isFull],
	["__WEBGL_MATERIAL_BASE_MAP_UNIFORMS__", material("baseMap")],
	["__WEBGL_MATERIAL_BASE_MAP_LEGACY_UNIFORMS__", all(isFull, material("baseMap"))],
	["__WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP_UNIFORMS__", all(isPBR, material("metallicRoughnessMap"))],
	["__WEBGL_MATERIAL_METALLIC_ROUGHNESS_MAP_LEGACY_UNIFORMS__", all(isFull, material("metallicRoughnessMap"))],
	["__WEBGL_MATERIAL_NORMAL_MAP_UNIFORMS__", all(isPBR, material("normalMap"))],
	["__WEBGL_MATERIAL_NORMAL_MAP_LEGACY_UNIFORMS__", all(isFull, material("normalMap"))],
	["__WEBGL_MATERIAL_EMISSIVE_MAP_UNIFORMS__", material("emissiveMap")],
	["__WEBGL_MATERIAL_EMISSIVE_MAP_LEGACY_UNIFORMS__", all(isFull, material("emissiveMap"))],
	["__WEBGL_MATERIAL_OCCLUSION_MAP_UNIFORMS__", all(isPBR, material("occlusionMap"))],
	["__WEBGL_MATERIAL_OCCLUSION_MAP_LEGACY_UNIFORMS__", all(isFull, material("occlusionMap"))],
	["__WEBGL_MATERIAL_IRIDESCENCE_MAP_UNIFORMS__", all(isPBR, material("iridescence"), material("iridescenceMap"))],
	["__WEBGL_MATERIAL_IRIDESCENCE_MAP_LEGACY_UNIFORMS__", all(isFull, material("iridescenceMap"))],
	["__WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP_UNIFORMS__", all(isPBR, material("iridescence"), material("iridescenceThicknessMap"))],
	["__WEBGL_MATERIAL_IRIDESCENCE_THICKNESS_MAP_LEGACY_UNIFORMS__", all(isFull, material("iridescenceThicknessMap"))],
	["__WEBGL_MATERIAL_ANISOTROPY_MAP_UNIFORMS__", all(isPBR, material("anisotropy"), material("anisotropyMap"))],
	["__WEBGL_MATERIAL_ANISOTROPY_MAP_LEGACY_UNIFORMS__", all(isFull, material("anisotropyMap"))],
	["__WEBGL_SCENE_ENVIRONMENT_SPECULAR_UNIFORMS__", all(isPBR, scene("environmentSpecular"))],
	["__WEBGL_SCENE_LOCAL_LIGHT_PROBE_UNIFORMS__", all(scene("sh"), scene("localLightProbes"))],
	["__WEBGL_IRRADIANCE_PROBE_GRID_UNIFORMS__", irradianceGrid],
	["__WEBGL_SCENE_REFLECTION_PROBE_UNIFORMS__", all(isPBR, scene("environmentSpecular"), scene("reflectionProbes"))],
	["__WEBGL_SCENE_FORWARD_LIGHT_UNIFORMS__", isLit],
	["__WEBGL_SCENE_SHADOW_UNIFORMS__", all(isLit, scene("shadows"))],
	["__WEBGL_SHADOW_TRANSMITTANCE_UNIFORMS__", shadowTransmittance],
	["__WEBGL_SCENE_CLUSTERED_LIGHT_UNIFORMS__", all(isLit, scene("clusteredLighting"))],
	["__WEBGL_SCENE_OIT_UNIFORMS__", parameter("specialization.oit")],
	["__WEBGL_SCENE_EXTRA_OUTPUTS__", any(parameter("specialization.oit"), equals("specialization.output", "mrt"))],
];

const uniformTemplate: ShaderSourceNode = {
	template: asset("scene.fragmentUniforms"),
	blocks: uniformMarkers.map(([start, condition], index) => ({
		start,
		end: uniformMarkers[index + 1]?.[0] ?? "__WEBGL_SCENE_TEMPLATE_END__",
		when: condition,
	})),
	replacements: [
		{
			marker: "__WEBGL_SCENE_TEMPLATE_END__",
			value: { defines: {}, sourcePath: "<webgl-scene-template-end>" },
		},
	],
};

const preludeTemplate: ShaderSourceNode = {
	template: asset("scene.fragmentPrelude"),
	replacements: [
		{
			marker: "__WEBGL_SHADOW_TRANSMITTANCE_DEFINE__",
			value: when(
				shadowTransmittance,
				{
					defines: { WEBGL_SHADOW_TRANSMITTANCE: literal(1) },
					sourcePath: "<webgl-shadow-transmittance-define>",
				},
			),
		},
	],
};

const localProbeTemplate: ShaderSourceNode = {
	template: asset("scene.fragmentLocalProbes"),
	replacements: [
		{
			marker: "__WEBGL_IRRADIANCE_PROBE_GRID_FUNCTIONS__",
			value: when(
				irradianceGrid,
				asset("internal.irradianceProbeGridFragment"),
				asset("internal.diffuseProbeFallbackFragment"),
			),
		},
	],
};

const shaderModules = createShaderModuleGroup(
	WEBGL_SHADER_FILES, "part.", "webgl.part.", "unknown",
);
const directiveModules = createShaderModuleGroup(
	WEBGL_DIRECTIVE_SHADER_FILES, "directive.", "webgl.directive.", "unknown",
);

const assets = {
	...shaderModules.assets,
	...directiveModules.assets,
	...mapShaderManifestEntries(WEBGL_SCENE_FRAGMENT_SHADER_FILES, "scene.",
		(path) => ({ path })),
	"internal.diffuseProbeFallbackFragment": {
		path: WEBGL_INTERNAL_SHADER_FILES.diffuseProbeFallbackFragment,
	},
	"internal.irradianceProbeGridFragment": {
		path: WEBGL_INTERNAL_SHADER_FILES.irradianceProbeGridFragment,
	},
	"fallback.localProbe": { path: "./webgl/scene/fragmentLocalProbeFallbacks.glsl" },
	"fallback.shadow": { path: "./webgl/scene/fragmentShadowFallbacks.glsl" },
	"fallback.environment": { path: "./webgl/scene/fragmentEnvironmentFallback.glsl" },
} as const;

const sceneVertex: ShaderSourceNode = {
	template: asset("part.sceneVertex"),
	replacements: [
		{
			marker: "__IGNIS_WEBGL_ANIMATION_DEFINES__",
			value: animationDefines(
				"specialization",
				{
					not: { equals: [parameter("specialization.morphSemanticMask"), literal(0)] },
				},
				"<webgl-scene-animation-defines>",
			),
		},
	],
};

const sources = {
	...shaderModules.sources,
	...directiveModules.sources,
	"webgl.scene": {
		kind: "program",
		sourceKind: "builtin-scene",
		parameters: sceneParams,
		stages: {
			vertex: sceneVertex,
			fragment: {
				concat: [
					preludeTemplate,
					sceneDefines,
					uniformTemplate,
					when(not(isFull), asset("scene.fragmentMaterialBlocks")),
					asset("scene.fragmentUvTextureNormal"),
					when(scene("sh"), asset("scene.fragmentSh")),
					when(all(scene("sh"), scene("localLightProbes")), localProbeTemplate),
					when(scene("clusteredLighting"), asset("scene.fragmentClusteredLighting")),
					when(scene("environmentSpecular"), asset("scene.fragmentEnvironmentSpecular")),
					when(all(scene("environmentSpecular"), scene("reflectionProbes")), asset("scene.fragmentReflectionProbes")),
					asset("scene.fragmentLightAttenuation"),
					when(scene("shadows"), asset("scene.fragmentShadows")),
					when(all(scene("sh"), not(scene("localLightProbes"))), asset("fallback.localProbe")),
					when(all(isLit, not(scene("shadows"))), asset("fallback.shadow")),
					when(all(scene("environmentSpecular"), not(scene("reflectionProbes"))), asset("fallback.environment")),
					when(isPBR, asset("scene.fragmentBrdfPbr")),
					when(isLegacy, asset("scene.fragmentPhong")),
					when(isPBR, asset("scene.fragmentPbrLighting")),
					asset("scene.fragmentMainOutput"),
				],
				fallbackSourcePath: "<webgl-scene-fragment-part>",
			},
		},
	},

	"webgl.scene.depth": {
		kind: "program",
		sourceKind: "builtin-scene",
		parameters: depthParams,
		stages: {
			vertex: {
				template: asset("part.sceneDepthPrepassVertex"),
				replacements: [{
					marker: "__IGNIS_WEBGL_ANIMATION_DEFINES__",
					value: animationDefines(
						"specialization",
						parameter("specialization.morphPosition"),
						"<webgl-depth-animation-defines>",
					),
				}],
			},
			fragment: {
				template: asset("part.sceneDepthPrepassFragment"),
				replacements: [{
					marker: "__IGNIS_WEBGL_DEPTH_DEFINES__",
					value: {
						defines: {
							WEBGL_DEPTH_ALPHA_MASK: parameter("specialization.alphaMask"),
							WEBGL_DEPTH_BASE_MAP: all(
								parameter("specialization.alphaMask"),
								parameter("specialization.baseMap"),
							),
						},
						sourcePath: "<webgl-depth-defines>",
					},
				}],
			},
		},
	},

	...mapShaderManifestEntries({
		"webgl.shadow.depth": "part.shadowDepthFragment",
		"webgl.shadow.transmittance": "part.shadowTransmittanceFragment",
	}, "", (fragment) => ({
		kind: "program",
		sourceKind: "shadow",
		parameters: shadowParams,
		stages: {
			vertex: {
				template: asset("part.shadowDepthVertex"),
				replacements: [{
					marker: "__IGNIS_WEBGL_ANIMATION_DEFINES__",
					value: animationDefines(
						"specialization",
						parameter("specialization.morphPosition"),
						"<webgl-shadow-animation-defines>",
					),
				}],
			},
			fragment: asset(fragment),
		},
	} as const)),
} as const;

export const WEBGL_SHADER_MANIFEST = {
	backend: "webgl",
	language: "glsl",
	assets,
	sources,
	preloadGroups: {
		backendInit: getShaderManifestSourceKeys(shaderModules.sources),
	},
	profile: {
		baseId: "ignis/webgl-profile-base",
		revision: 1,
		includes: [
			{ id: "ignis/webgl/animation.glsl", source: "webgl.directive.animation" },
			{ id: "ignis/webgl/constants-base.glsl", source: "webgl.directive.constants" },
			{ id: "ignis/color/srgb.glsl", source: "webgl.directive.srgb" },
			{ id: "ignis/postprocess/luma-weights.glsl", source: "webgl.directive.lumaWeights" },
			{ id: "ignis/postprocess/luma-common.glsl", source: "webgl.directive.lumaCommon" },
		],
		overlay: {
			id: "ignis/webgl-instance-overlay",
			includeId: "ignis/webgl/constants.glsl",
			sourcePath: "runtime://ignis/includes/glsl/webgl/constants.glsl",
			baseInclude: "ignis/webgl/constants-base",
			parameters: {
				type: "record",
				fields: Object.fromEntries(
					[
						"maxDirectionalLights", "maxPointLights", "maxSpotLights",
						"maxClusterLightsPerFragment", "maxLocalLightProbes",
						"maxReflectionProbes",
					].map((name) => [name, { type: "integer", required: true, min: 0 }]),
				),
			},
			defines: {
				__WEBGL_MAX_DIRECTIONAL_LIGHTS__: parameter("maxDirectionalLights"),
				__WEBGL_MAX_POINT_LIGHTS__: parameter("maxPointLights"),
				__WEBGL_MAX_SPOT_LIGHTS__: parameter("maxSpotLights"),
				__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__: parameter("maxClusterLightsPerFragment"),
				__WEBGL_MAX_LOCAL_LIGHT_PROBES__: parameter("maxLocalLightProbes"),
				__WEBGL_MAX_REFLECTION_PROBES__: parameter("maxReflectionProbes"),
			},
		},
	},
} as const satisfies ShaderBackendManifest<typeof sources>;
