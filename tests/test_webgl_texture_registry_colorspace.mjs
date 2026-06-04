import assert from "node:assert/strict";
import { Texture } from "../src/core/Texture.ts";
import { WebGLTextureRegistry } from "../src/renderers/webgl/WebGLTextureRegistry.ts";
import { TextureFormat } from "../src/renderers/types.ts";

function createTextureRegistryTestGL(options = {}) {
	let textureId = 0;
	const textureParameterCalls = [];
	const texImage2DCalls = [];
	let generateMipmapCallCount = 0;
	const gl = {
		MAX_TEXTURE_SIZE: 0x0d33,
		TEXTURE_2D: 0x0de1,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_BASE_LEVEL: 0x813c,
		TEXTURE_MAX_LEVEL: 0x813d,
		UNPACK_ALIGNMENT: 0x0cf5,
		LINEAR: 0x2601,
		NEAREST: 0x2600,
		NEAREST_MIPMAP_NEAREST: 0x2700,
		LINEAR_MIPMAP_NEAREST: 0x2701,
		NEAREST_MIPMAP_LINEAR: 0x2702,
		LINEAR_MIPMAP_LINEAR: 0x2703,
		REPEAT: 0x2901,
		CLAMP_TO_EDGE: 0x812f,
		MIRRORED_REPEAT: 0x8370,
		RGBA: 0x1908,
		RED: 0x1903,
		RG: 0x8227,
		R8: 0x8229,
		RG8: 0x822b,
		R16F: 0x822d,
		RG16F: 0x822f,
		R32F: 0x822e,
		RG32F: 0x8230,
		RGBA16F: 0x881a,
		RGBA32F: 0x8814,
		SRGB8_ALPHA8: 0x8c43,
		UNSIGNED_BYTE: 0x1401,
		HALF_FLOAT: 0x140b,
		FLOAT: 0x1406,
		getParameter(parameter) {
			if (parameter === this.MAX_TEXTURE_SIZE) {
				return 4096;
			}
			return 0;
		},
		getExtension(name) {
			if (
				name === "OES_texture_float_linear" ||
				name === "OES_texture_half_float_linear"
			) {
				return options.floatLinearExtension ? {} : null;
			}
			return null;
		},
		createTexture() {
			return { id: `tex-${++textureId}` };
		},
		deleteTexture() {},
		bindTexture() {},
		pixelStorei() {},
		texParameteri(target, pname, value) {
			textureParameterCalls.push({ target, pname, value });
		},
		texImage2D(
			target,
			level,
			internalFormat,
			width,
			height,
			border,
			format,
			type,
			pixels
		) {
			texImage2DCalls.push({
				target,
				level,
				internalFormat,
				width,
				height,
				border,
				format,
				type,
				pixels,
			});
		},
		generateMipmap() {
			generateMipmapCallCount++;
		},
		textureParameterCalls,
		texImage2DCalls,
		get generateMipmapCallCount() {
			return generateMipmapCallCount;
		},
	};
	if (options.rgba16f === false) {
		delete gl.RGBA16F;
		delete gl.HALF_FLOAT;
	}
	if (options.rgba32f === false) {
		delete gl.RGBA32F;
		delete gl.FLOAT;
	}
	return gl;
}

function createSolidTexture(colorSpace) {
	return new Texture(new Uint8Array([128, 64, 32, 255]), 1, 1, colorSpace);
}

function testEnvironmentTextureRespectsTextureColorSpace() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});

	const srgbTexture = createSolidTexture("sRGB");
	const linearTexture = createSolidTexture("Linear");
	const hdrTexture = createSolidTexture("HDR");

	const srgbResolvedA = registry.getEnvironmentTexture(srgbTexture);
	const srgbResolvedB = registry.getEnvironmentTexture(srgbTexture);
	const linearResolved = registry.getEnvironmentTexture(linearTexture);
	const hdrResolved = registry.getEnvironmentTexture(hdrTexture);

	assert.equal(srgbResolvedA.isLinear, false);
	assert.equal(srgbResolvedB.isLinear, false);
	assert.equal(linearResolved.isLinear, true);
	assert.equal(hdrResolved.isLinear, true);
}

function testBaseColorTextureRemainsSrgbByDefault() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});

	const srgbTexture = createSolidTexture("sRGB");
	const linearTexture = createSolidTexture("Linear");

	assert.equal(registry.getBaseColorTexture(srgbTexture).isLinear, false);
	assert.equal(registry.getBaseColorTexture(linearTexture).isLinear, true);
}

function testEnvironmentTextureLimitsMaxMipLevelToUploadedChain() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});

	const envTexture = new Texture(
		new Float32Array(128 * 64 * 4),
		128,
		64,
		"HDR"
	);
	envTexture.mipmaps = [
		new Float32Array(128 * 64 * 4),
		new Float32Array(64 * 32 * 4),
		new Float32Array(32 * 16 * 4),
		new Float32Array(16 * 8 * 4),
		new Float32Array(8 * 4 * 4),
	];
	envTexture.data = envTexture.mipmaps[0];

	registry.getEnvironmentSpecularTexture(envTexture);

	const maxLevelCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MAX_LEVEL
	);
	assert.ok(maxLevelCall);
	assert.equal(maxLevelCall.value, envTexture.mipmaps.length - 1);
}

function testMipmapFilterGeneratesMipChainWhenOnlyBaseLevelExists() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture(new Uint8Array(4 * 4 * 4), 4, 4, "sRGB");
	texture.minFilter = "LinearMipmapLinear";

	registry.getBaseColorTexture(texture);

	const maxLevelCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MAX_LEVEL
	);
	const minFilterCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MIN_FILTER
	);
	assert.ok(maxLevelCall);
	assert.ok(minFilterCall);
	assert.equal(maxLevelCall.value, 2);
	assert.equal(minFilterCall.value, gl.LINEAR_MIPMAP_LINEAR);
	assert.equal(gl.generateMipmapCallCount, 1);
}

function testLinearFilterSkipsMipmapGenerationForSingleLevelTexture() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture(new Uint8Array(4 * 4 * 4), 4, 4, "sRGB");
	texture.minFilter = "Linear";

	registry.getBaseColorTexture(texture);

	const maxLevelCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MAX_LEVEL
	);
	const minFilterCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MIN_FILTER
	);
	assert.ok(maxLevelCall);
	assert.ok(minFilterCall);
	assert.equal(maxLevelCall.value, 0);
	assert.equal(minFilterCall.value, gl.LINEAR);
	assert.equal(gl.generateMipmapCallCount, 0);
}

function testNearestMipmapLinearMapsToNearestMipmapLinear() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture(new Uint8Array(4 * 4 * 4), 4, 4, "sRGB");
	texture.mipmaps = [
		new Uint8Array(4 * 4 * 4),
		new Uint8Array(2 * 2 * 4),
	];
	texture.data = texture.mipmaps[0];
	texture.minFilter = "NearestMipmapLinear";

	registry.getBaseColorTexture(texture);

	const minFilterCall = gl.textureParameterCalls.find(
		(call) => call.pname === gl.TEXTURE_MIN_FILTER
	);
	assert.ok(minFilterCall);
	assert.equal(minFilterCall.value, gl.NEAREST_MIPMAP_LINEAR);
	assert.equal(gl.generateMipmapCallCount, 0);
}

function testEnvironmentSpecularFloatMipChainUploadsAsRGBA16F() {
	const gl = createTextureRegistryTestGL({ floatLinearExtension: true });
	const registry = new WebGLTextureRegistry(gl, () => {});
	const envTexture = new Texture(null, 2, 1, "HDR");
	envTexture.mipmaps = [
		new Float32Array([2, 0.5, 0.25, 1, 4, 2, 1, 1]),
		new Float32Array([3, 1.5, 0.75, 1]),
	];
	envTexture.data = envTexture.mipmaps[0];

	registry.getEnvironmentSpecularTexture(envTexture);

	assert.equal(gl.texImage2DCalls.length, 2);
	assert.equal(gl.texImage2DCalls[0].internalFormat, gl.RGBA16F);
	assert.equal(gl.texImage2DCalls[0].format, gl.RGBA);
	assert.equal(gl.texImage2DCalls[0].type, gl.HALF_FLOAT);
	assert.ok(gl.texImage2DCalls[0].pixels instanceof Uint16Array);
	assert.equal(gl.texImage2DCalls[0].pixels[0], 0x4000);
	assert.equal(gl.texImage2DCalls[0].pixels[3], 0x3c00);
	assert.equal(gl.texImage2DCalls[1].internalFormat, gl.RGBA16F);
	assert.ok(gl.texImage2DCalls[1].pixels instanceof Uint16Array);
}

function testEnvironmentSpecularFloatUploadFallsBackToRGBA32F() {
	const gl = createTextureRegistryTestGL({
		rgba16f: false,
		floatLinearExtension: true,
	});
	const registry = new WebGLTextureRegistry(gl, () => {});
	const envTexture = new Texture(
		new Float32Array([2, 0.5, 0.25, 1]),
		1,
		1,
		"HDR"
	);

	registry.getEnvironmentSpecularTexture(envTexture);

	assert.equal(gl.texImage2DCalls.length, 1);
	assert.equal(gl.texImage2DCalls[0].internalFormat, gl.RGBA32F);
	assert.equal(gl.texImage2DCalls[0].type, gl.FLOAT);
	assert.ok(gl.texImage2DCalls[0].pixels instanceof Float32Array);
	assert.equal(gl.texImage2DCalls[0].pixels[0], 2);
}

function testEnvironmentSpecularKeepsFloatCacheSeparateFromBaseColor() {
	const gl = createTextureRegistryTestGL({ floatLinearExtension: true });
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture(
		new Float32Array([2, 0.5, 0.25, 1]),
		1,
		1,
		"HDR"
	);

	registry.getBaseColorTexture(texture);
	registry.getEnvironmentSpecularTexture(texture);

	assert.equal(gl.texImage2DCalls.length, 2);
	assert.equal(gl.texImage2DCalls[0].internalFormat, gl.RGBA);
	assert.equal(gl.texImage2DCalls[0].type, gl.UNSIGNED_BYTE);
	assert.ok(gl.texImage2DCalls[0].pixels instanceof Uint8Array);
	assert.equal(gl.texImage2DCalls[0].pixels[0], 255);
	assert.equal(gl.texImage2DCalls[1].internalFormat, gl.RGBA16F);
	assert.equal(gl.texImage2DCalls[1].type, gl.HALF_FLOAT);
	assert.ok(gl.texImage2DCalls[1].pixels instanceof Uint16Array);
}

function testExplicitR8TextureUploadsSingleChannel() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture({
		width: 2,
		height: 1,
		format: TextureFormat.R8Unorm,
		colorSpace: "Linear",
		data: new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]),
	});

	registry.getBaseColorTexture(texture);

	assert.equal(gl.texImage2DCalls.length, 1);
	assert.equal(gl.texImage2DCalls[0].internalFormat, gl.R8);
	assert.equal(gl.texImage2DCalls[0].format, gl.RED);
	assert.deepEqual(Array.from(gl.texImage2DCalls[0].pixels), [10, 50]);
}

function testExplicitSrgbTextureUsesHardwareDecode() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});
	const texture = new Texture({
		width: 1,
		height: 1,
		format: TextureFormat.RGBA8UnormSrgb,
		colorSpace: "sRGB",
		data: new Uint8Array([128, 64, 32, 255]),
	});

	const resolved = registry.getBaseColorTexture(texture);

	assert.equal(resolved.isLinear, true);
	assert.equal(gl.texImage2DCalls[0].internalFormat, gl.SRGB8_ALPHA8);
	assert.equal(gl.texImage2DCalls[0].format, gl.RGBA);
}

function run() {
	testEnvironmentTextureRespectsTextureColorSpace();
	testBaseColorTextureRemainsSrgbByDefault();
	testEnvironmentTextureLimitsMaxMipLevelToUploadedChain();
	testMipmapFilterGeneratesMipChainWhenOnlyBaseLevelExists();
	testLinearFilterSkipsMipmapGenerationForSingleLevelTexture();
	testNearestMipmapLinearMapsToNearestMipmapLinear();
	testEnvironmentSpecularFloatMipChainUploadsAsRGBA16F();
	testEnvironmentSpecularFloatUploadFallsBackToRGBA32F();
	testEnvironmentSpecularKeepsFloatCacheSeparateFromBaseColor();
	testExplicitR8TextureUploadsSingleChannel();
	testExplicitSrgbTextureUsesHardwareDecode();
	console.log("WebGL texture registry color-space tests passed");
}

run();
