import assert from "node:assert/strict";
import { Texture } from "../src/core/Texture.ts";
import { WebGLTextureRegistry } from "../src/renderers/webgl/WebGLTextureRegistry.ts";

function createTextureRegistryTestGL() {
	let textureId = 0;
	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		TEXTURE_2D: 0x0de1,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_MIN_FILTER: 0x2801,
		UNPACK_ALIGNMENT: 0x0cf5,
		LINEAR: 0x2601,
		NEAREST: 0x2600,
		NEAREST_MIPMAP_NEAREST: 0x2700,
		LINEAR_MIPMAP_LINEAR: 0x2703,
		REPEAT: 0x2901,
		CLAMP_TO_EDGE: 0x812f,
		MIRRORED_REPEAT: 0x8370,
		RGBA: 0x1908,
		UNSIGNED_BYTE: 0x1401,
		getParameter(parameter) {
			if (parameter === this.MAX_TEXTURE_SIZE) {
				return 4096;
			}
			return 0;
		},
		createTexture() {
			return { id: `tex-${++textureId}` };
		},
		deleteTexture() {},
		bindTexture() {},
		pixelStorei() {},
		texParameteri() {},
		texImage2D() {},
		generateMipmap() {},
	};
}

function createSolidTexture(colorSpace) {
	return new Texture(new Uint8Array([128, 64, 32, 255]), 1, 1, colorSpace);
}

function testSkyboxTextureRespectsTextureColorSpace() {
	const gl = createTextureRegistryTestGL();
	const registry = new WebGLTextureRegistry(gl, () => {});

	const srgbTexture = createSolidTexture("sRGB");
	const linearTexture = createSolidTexture("Linear");
	const hdrTexture = createSolidTexture("HDR");

	const srgbResolvedA = registry.getSkyboxTexture(srgbTexture);
	const srgbResolvedB = registry.getSkyboxTexture(srgbTexture);
	const linearResolved = registry.getSkyboxTexture(linearTexture);
	const hdrResolved = registry.getSkyboxTexture(hdrTexture);

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

function run() {
	testSkyboxTextureRespectsTextureColorSpace();
	testBaseColorTextureRemainsSrgbByDefault();
	console.log("WebGL texture registry color-space tests passed");
}

run();
