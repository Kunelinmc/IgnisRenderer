import { expect, test } from "@playwright/test";

test("WebGL compiles built-in and ShaderMaterial deformation ABI", async ({ page }) => {
	await page.goto("/");
	const result = await page.evaluate(async () => {
		const canvas = document.createElement("canvas");
		const gl = canvas.getContext("webgl2");
		if (!gl) return { supported: false as const, errors: [] as string[] };
		const { ShaderSource } = await import("/src/shaders/ShaderSource.ts");
		const {
			DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			ShaderBackendCompileStage,
			ShaderRuntime,
		} = await import("/src/shaders/runtime/index.ts");
		const stage = new ShaderBackendCompileStage({
			backend: "webgl",
			runtime: new ShaderRuntime({ mode: "strict" }),
			profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			mode: "strict",
		});
		const limits = {
			maxDirectionalLights: 4,
			maxPointLights: 16,
			maxSpotLights: 8,
		};
		const variant = {
			output: "single" as const,
			materialGBuffer: false,
			oit: false,
			scene: {
				shadows: false,
				shadowTransmittance: false,
				clusteredLighting: false,
				sh: false,
				localLightProbes: false,
				irradianceProbeGrid: false,
				reflectionProbes: false,
				environmentSpecular: false,
			},
			material: {
				model: "unlit" as const,
				baseMap: false,
				metallicRoughnessMap: false,
				specularMap: false,
				specularColorMap: false,
				normalMap: false,
				emissiveMap: false,
				occlusionMap: false,
				clearcoat: false,
				clearcoatMap: false,
				clearcoatRoughnessMap: false,
				clearcoatNormalMap: false,
				sheen: false,
				sheenColorMap: false,
				sheenRoughnessMap: false,
				iridescence: false,
				iridescenceMap: false,
				iridescenceThicknessMap: false,
				anisotropy: false,
				anisotropyMap: false,
				transmission: false,
				transmissionMap: false,
				thicknessMap: false,
				alphaMask: false,
			},
			skinProfile: "skin8" as const,
			morphSemanticMask: 3,
		};
		const scene = await ShaderSource.load("webgl.scene.composite", {
			limits,
			variant,
		});
		const shadow = await ShaderSource.load(
			"webgl.part.shadowDepthVertex.composite",
		);
		const sources = [
			{
				label: "scene",
				code: scene.vertex.code,
				sourceMap: scene.vertex.sourceMap,
			},
			{
				label: "shadow",
				code: shadow.code.replace(
					"__IGNIS_WEBGL_ANIMATION_DEFINES__",
					"#define IGNIS_WEBGL_DEFORMATION_ACTIVE 1\n" +
						"#define IGNIS_WEBGL_SKIN_INFLUENCES 8",
				),
				sourceMap: shadow.sourceMap,
			},
			{
				label: "custom-material",
				code: `#version 300 es
precision highp float;
#import <ignis/webgl/animation>
void main() { gl_Position = vec4(0.0); }`,
				sourceMap: null,
			},
		];
		const errors: string[] = [];
		for (const source of sources) {
			const processed = await stage.compileAsync({
				code: source.code,
				language: "glsl",
				stage: "vertex",
				entryPoint: "main",
				label: source.label,
				sourceKind: source.label === "custom-material" ?
					"custom-material" : source.label === "shadow" ? "shadow" : "builtin-scene",
				sourceMap: source.sourceMap,
				directiveSourcePath: source.label,
			});
			const shader = gl.createShader(gl.VERTEX_SHADER)!;
			gl.shaderSource(shader, processed.code);
			gl.compileShader(shader);
			if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
				errors.push(`${source.label}: ${gl.getShaderInfoLog(shader)}`);
			}
			gl.deleteShader(shader);
		}
		return { supported: true as const, errors };
	});

	test.skip(!result.supported, "WebGL2 is unavailable.");
	if (!result.supported) return;
	expect(result.errors).toEqual([]);
});
