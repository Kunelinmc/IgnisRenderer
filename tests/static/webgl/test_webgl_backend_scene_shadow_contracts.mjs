import assert from "node:assert/strict";import { Material } from "../../../src/materials/Material.ts";import { Logger } from "../../../src/foundation/Logger.ts";import { WebGLShadowRasterPass } from "../../../src/backends/webgl/WebGLShadowRasterPass.ts";import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";import { TEST_SCENE_LIMITS, getTestSceneShader, createShadowRasterCaptureGL, createShadowPassHost, createShadowRasterPlan, createShadowPacket, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

function testSceneShaderBackLitShadowGuard() {
	const shader = getTestSceneShader();
	const doubleSidedNormalFlip = "if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)";
	const firstFlipIndex = shader.fragment.indexOf(doubleSidedNormalFlip);
	const secondFlipIndex = shader.fragment.indexOf(
		doubleSidedNormalFlip,
		firstFlipIndex + doubleSidedNormalFlip.length,
	);
	const normalMapIndex = shader.fragment.indexOf("normal = applyNormalMap(");
	assert.ok(shader.fragment.includes("dot(normal, lightDirection) <= 0.0"));
	assert.ok(shader.fragment.includes("uniform int uDoubleSided;"));
	assert.ok(firstFlipIndex >= 0);
	assert.ok(secondFlipIndex > normalMapIndex);
}

function testSceneShaderKeepsClusteredFragmentLightLimitPlaceholder() {
	const shader = getTestSceneShader();
	assert.ok(shader.fragment.includes("__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__"));
}

function testSceneShaderUsesFlippedShadowNormal() {
	const shader = getTestSceneShader();
	const doubleSidedNormalFlip = "if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)";
	const firstFlipIndex = shader.fragment.indexOf(doubleSidedNormalFlip);
	const secondFlipIndex = shader.fragment.indexOf(
		doubleSidedNormalFlip,
		firstFlipIndex + doubleSidedNormalFlip.length,
	);
	const shadowNormalIndex = shader.fragment.indexOf("vec3 shadowNormal = normal;");
	assert.ok(firstFlipIndex >= 0);
	assert.ok(secondFlipIndex > firstFlipIndex);
	assert.ok(shadowNormalIndex > secondFlipIndex);
	assert.ok(/shadePBR\(\s*albedo,\s*normal,\s*shadowNormal,\s*viewDir,/.test(shader.fragment));
	assert.ok(shader.fragment.includes("shadePhong(albedo, normal, shadowNormal, viewDir);"));
	assert.ok(/sampleDirectionalShadowVisibility\([\s\S]*shadowNormal/.test(shader.fragment));
	assert.ok(shader.fragment.includes("uDirShadowParamsC"));
	assert.ok(shader.fragment.includes("uDirShadowCascadeViewProjection"));
	assert.ok(shader.fragment.includes("uDirShadowCascadeSplits"));
	assert.ok(shader.fragment.includes("resolveDirectionalCascadeIndex"));
	assert.ok(shader.fragment.includes("uSpotShadowParamsC"));
	assert.ok(shader.fragment.includes("uParticleShadowVolumeAtlas"));
	assert.ok(shader.fragment.includes("uParticleShadowVolumeSliceParams"));
	assert.ok(shader.fragment.includes("sampleParticleShadowVolumeTransmittance"));
}

function testShadowRasterPassConsumesResolvedPlanAndRestoresBaseline() {
	const gl = createShadowRasterCaptureGL();
	const host = createShadowPassHost(gl);
	const pass = new WebGLShadowRasterPass(host);
	const material = new Material({
		doubleSided: false,
		cullMode: "front",
	});
	const packet = createShadowPacket(material);
	const plan = createShadowRasterPlan({
		casterPackets: [packet],
		transmitterPackets: [packet],
	});

	const prepared = pass.prepare(plan);
	assert.equal(prepared.atlasTileSize, 64);
	gl.calls.disable.length = 0;
	gl.calls.viewport.length = 0;
	pass.render(plan);

	assert.equal(host.cullModeCalls, 0);
	assert.equal(gl.calls.disable.filter((capability) => capability === gl.CULL_FACE).length, 3);
	assert.equal(gl.calls.drawElements.length, 2);
	assert.deepEqual(gl.calls.viewport[0], { x: 8, y: 16, width: 32, height: 32 });
	assert.deepEqual(gl.calls.viewport.at(-1), { x: 0, y: 0, width: 320, height: 180 });
	pass.destroy();
}

function testShadowRasterPassRestoresDefaultFramebufferDrawBuffer() {
	const gl = createShadowRasterCaptureGL();
	const operations = [];
	const originalBindFramebuffer = gl.bindFramebuffer;
	const originalDrawBuffers = gl.drawBuffers;
	gl.bindFramebuffer = (target, framebuffer) => {
		operations.push(["bindFramebuffer", framebuffer]);
		originalBindFramebuffer(target, framebuffer);
	};
	gl.drawBuffers = (buffers) => {
		operations.push(["drawBuffers", [...buffers]]);
		originalDrawBuffers(buffers);
	};
	const pass = new WebGLShadowRasterPass(createShadowPassHost(gl));
	const plan = createShadowRasterPlan({ baselineFramebuffer: null });

	pass.prepare(plan);
	assert.deepEqual(operations.slice(-3), [
		["bindFramebuffer", null],
		["drawBuffers", [gl.BACK]],
		["bindFramebuffer", null],
	]);
	operations.length = 0;
	pass.render(plan);
	assert.deepEqual(operations.slice(-3), [
		["bindFramebuffer", null],
		["drawBuffers", [gl.BACK]],
		["bindFramebuffer", null],
	]);
	pass.destroy();
}

function testShadowRasterPassCleansPartialAllocationAndRestoresOnDrawError() {
	const incompleteGL = createShadowRasterCaptureGL();
	incompleteGL.checkFramebufferStatus = () => 0x8cd6;
	const incompletePass = new WebGLShadowRasterPass(createShadowPassHost(incompleteGL));
	assert.throws(
		() => incompletePass.prepare(createShadowRasterPlan()),
		/WebGL shadow framebuffer is incomplete/,
	);
	assert.equal(incompleteGL.calls.deletedTextures.length, 2);
	assert.equal(incompleteGL.calls.deletedFramebuffers.length, 1);
	assert.deepEqual(incompleteGL.calls.viewport.at(-1), {
		x: 0,
		y: 0,
		width: 320,
		height: 180,
	});

	const throwingGL = createShadowRasterCaptureGL();
	const throwingPass = new WebGLShadowRasterPass(createShadowPassHost(throwingGL));
	const packet = createShadowPacket(new Material());
	const plan = createShadowRasterPlan({ casterPackets: [packet] });
	throwingPass.prepare(plan);
	throwingGL.calls.viewport.length = 0;
	throwingGL.drawElements = () => {
		throw new Error("draw failed");
	};
	assert.throws(() => throwingPass.render(plan), /draw failed/);
	assert.deepEqual(throwingGL.calls.viewport.at(-1), {
		x: 0,
		y: 0,
		width: 320,
		height: 180,
	});
	throwingPass.destroy();
}

function testShadowRasterPassKeepsSkinningDiagnosticKey() {
	const gl = createShadowRasterCaptureGL();
	const pass = new WebGLShadowRasterPass(createShadowPassHost(gl));
	const packet = createShadowPacket(new Material());
	packet.meshInstance.skeleton = {};
	const warnings = [];
	Logger.configure({
		level: "warn",
		resetOnceKeys: true,
		sink: {
			warn(...args) {
				warnings.push(args.join(" "));
			},
		},
	});
	try {
		const plan = createShadowRasterPlan({ casterPackets: [packet] });
		pass.prepare(plan);
		pass.render(plan);
	} finally {
		pass.destroy();
		Logger.reset();
	}
	assert.ok(warnings.some((warning) => warning.includes("[webgl-shadow-skinning-unsupported]")));
}

function testSceneShaderIncludesReflectionProbeUniforms() {
	const shader = getTestSceneShader();
	assert.ok(shader.fragment.includes("uniform sampler2D uEnvSpecularMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uBrdfLUT;"));
	assert.ok(shader.fragment.includes("uReflectionProbeCount"));
	assert.ok(shader.fragment.includes("uReflectionProbeWorldToProbeRow0"));
	assert.ok(shader.fragment.includes("computeReflectionProbeParallaxDirection"));
	assert.ok(shader.fragment.includes("computeReflectionProbeDepthOcclusion"));
	assert.ok(shader.fragment.includes("sampleEnvironmentSpecular"));
	assert.ok(shader.fragment.includes("uniform vec4 uTransmissionVolume;"));
	assert.ok(shader.fragment.includes("uniform vec4 uAttenuationColor;"));
	assert.ok(shader.fragment.includes("uniform vec4 uIridescence;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uIridescenceMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uIridescenceThicknessMap;"));
	assert.ok(shader.fragment.includes("float ior = max(uTransmissionVolume.x, 1.0);"));
	assert.ok(shader.fragment.includes("volumeAttenuation = exp(-absorb * thickness);"));
	assert.ok(shader.fragment.includes("refract(-viewDir, refractNormal, eta)"));
	assert.ok(shader.fragment.includes("resolveIridescenceFresnel"));
	assert.ok(shader.fragment.includes("diffuseFresnelWeight"));
}

function testSceneShaderIncludesLocalizedLightProbeUniforms() {
	const shader = getTestSceneShader();
	assert.ok(shader.fragment.includes("uniform int uLocalLightProbeCount;"));
	assert.ok(shader.fragment.includes("uLocalLightProbeWorldToProbeRow0"));
	assert.ok(shader.fragment.includes("uLocalLightProbeCoeffs"));
	assert.ok(shader.fragment.includes("selectTopTwoLocalLightProbes"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeIrradiance"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeRadiance"));
}

function testFullSceneShaderDeclaresExtensionSamplersForDynamicLayout() {
	const shader = getTestSceneShader();
	const samplerMatches = shader.fragment.match(/\buniform\s+sampler2D\b/g) ?? [];

	assert.equal(samplerMatches.length, 30);
	assert.ok(shader.fragment.includes("uniform sampler2D uAnisotropyMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uClearcoatMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uTransmissionBackgroundMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uTransmissionDepthMap;"));
	assert.ok(shader.fragment.includes("uniform vec3 uSHAmbientCoeffs[SH_COEFFICIENT_COUNT];"));
	assert.ok(!shader.fragment.includes("uniform sampler2D uSHAmbientCoeffs;"));
	assert.ok(shader.fragment.includes("uIrradianceProbeGridCoeffs"));
	assert.ok(shader.fragment.includes("vec3 sampleDiffuseProbeIrradiance"));
	assert.ok(shader.fragment.includes("sampleIrradianceProbeGridIrradiance"));
}

function testSceneShaderIncludesIrradianceProbeGridWhenEnabled() {
	const shader = ShaderSource.get("webgl.scene.raw", {
		limits: TEST_SCENE_LIMITS,
	});
	const samplerMatches = shader.fragment.match(/\buniform\s+sampler2D\b/g) ?? [];

	assert.equal(samplerMatches.length, 30);
	assert.ok(shader.fragment.includes("uniform sampler2D uIrradianceProbeGridCoeffs;"));
	assert.ok(shader.fragment.includes("uIrradianceProbeGridWorldToGridRow0"));
	assert.ok(shader.fragment.includes("sampleIrradianceProbeGridIrradiance"));
	assert.ok(shader.fragment.includes("return mix(fallback, gridAmbientBase.rgb"));
}

function testSceneShaderIncludesPBRTextureAndUV1Pipeline() {
	const shader = getTestSceneShader();
	assert.ok(shader.vertex.includes("layout(location = 3) in vec2 aUv1;"));
	assert.ok(shader.vertex.includes("layout(location = 6) in vec4 aTangent;"));
	assert.ok(shader.vertex.includes("out vec2 vUv1;"));
	assert.ok(shader.vertex.includes("out vec4 vTangent;"));
	assert.ok(shader.fragment.includes("in vec2 vUv1;"));
	assert.ok(shader.fragment.includes("in vec4 vTangent;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uMetallicRoughnessMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uNormalMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uOcclusionMap;"));
	assert.ok(shader.fragment.includes("uniform int uBaseMapUV;"));
	assert.ok(shader.fragment.includes("vec2 resolveUV(int uvSet) {"));
	assert.ok(shader.fragment.includes("uniform vec4 uBaseMapTransformA;"));
	assert.ok(shader.fragment.includes("uniform vec2 uBaseMapTransformB;"));
	assert.ok(shader.fragment.includes("vec2 resolveMappedUV("));
	assert.ok(shader.fragment.includes("bool resolveTangentFrame("));
	assert.ok(shader.fragment.includes("vec4 tangent,"));
}

function testSceneShaderIncludesOITPassMode() {
	const shader = getTestSceneShader();

	assert.ok(shader.fragment.includes("uniform int uOITPassMode;"));
	assert.ok(shader.fragment.includes("float resolveOITWeight("));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 1)"));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 2)"));
}

function testParticleShaderIncludesOITPassMode() {
	const shader = ShaderSource.get("webgl.part.particleFragment.raw");

	assert.ok(shader.includes("uniform int uOITPassMode;"));
	assert.ok(shader.includes("float resolveParticleOITWeight("));
	assert.ok(shader.includes("if (uOITPassMode == 1)"));
	assert.ok(shader.includes("if (uOITPassMode == 2)"));
}

await runWebGLBackendFile(
	[
		testSceneShaderBackLitShadowGuard,
		testSceneShaderKeepsClusteredFragmentLightLimitPlaceholder,
		testSceneShaderUsesFlippedShadowNormal,
		testShadowRasterPassConsumesResolvedPlanAndRestoresBaseline,
		testShadowRasterPassRestoresDefaultFramebufferDrawBuffer,
		testShadowRasterPassCleansPartialAllocationAndRestoresOnDrawError,
		testShadowRasterPassKeepsSkinningDiagnosticKey,
		testSceneShaderIncludesReflectionProbeUniforms,
		testSceneShaderIncludesLocalizedLightProbeUniforms,
	testFullSceneShaderDeclaresExtensionSamplersForDynamicLayout,
		testSceneShaderIncludesIrradianceProbeGridWhenEnabled,
		testSceneShaderIncludesPBRTextureAndUV1Pipeline,
		testSceneShaderIncludesOITPassMode,
		testParticleShaderIncludesOITPassMode,
	],
	"WebGL scene and shadow contract tests",
);
