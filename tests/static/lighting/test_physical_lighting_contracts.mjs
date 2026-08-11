import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";

const ROOT = new URL("../../../", import.meta.url);

function readSource(path) {
	return readFileSync(new URL(path, ROOT), "utf8");
}

function testPhongDefaultAndNormalizedEquation() {
	const material = new PhongMaterial();
	assert.deepEqual(material.specular, { r: 56, g: 56, b: 56 });

	const source = readSource("src/shaders/webgl/scene/fragmentPhong.glsl");
	assert.match(source, /\(shininess \+ 8\.0\) \/ \(8\.0 \* PI\)/);
	assert.match(source, /f0 \+ \(vec3\(1\.0\) - f0\) \* pow\(1\.0 - vDotH, 5\.0\)/);
	assert.match(source, /\(vec3\(1\.0\) - fresnel\) \* albedo \/ PI/);
	assert.doesNotMatch(source, /specular\s*\*\s*0\.25/);
}

function testPBRShaderContracts() {
	const output = readSource("src/shaders/webgl/scene/fragmentMainOutput.glsl");
	const lighting = readSource("src/shaders/webgl/scene/fragmentPbrLighting.glsl");
	assert.match(output, /texture\(uSpecularMap[^;]+\.a/);
	assert.match(output, /texture\(uClearcoatMap[^;]+\.r/);
	assert.match(output, /texture\(uClearcoatRoughnessMap[^;]+\.g/);
	assert.match(output, /texture\(uSheenRoughnessMap[^;]+\.a/);
	assert.match(output, /texture\(uTransmissionMap[^;]+\.r/);
	assert.match(output, /texture\(uThicknessMap[^;]+\.g/);
	assert.match(output, /texture\(uAnisotropyMap/);
	assert.doesNotMatch(output, /uIridescenceThicknessMap[^\n]+anisotropy/i);
	assert.match(lighting, /iridescence > EPSILON \? ambientFresnel : f0/);
	assert.doesNotMatch(lighting, /vec3\(0\.05\)/);
}

function testPhongSpecularLobeNormalization() {
	for (const shininess of [0, 1, 8, 32, 128, 512]) {
		for (const f0 of [0.04, 0.5]) {
			let integral = 0;
			const steps = 20000;
			for (let index = 0; index < steps; index++) {
				const nDotL = (index + 0.5) / steps;
				const nDotH = Math.sqrt((1 + nDotL) * 0.5);
				const fresnel = f0 + (1 - f0) * Math.pow(1 - nDotH, 5);
				const brdf = fresnel * (shininess + 8) / (8 * Math.PI) *
					Math.pow(nDotH, shininess);
				integral += brdf * nDotL * 2 * Math.PI / steps;
			}
			assert.ok(
				integral <= 1 + 1e-3,
				`shininess=${shininess}, F0=${f0}, energy=${integral}`,
			);
		}
	}
}

testPhongDefaultAndNormalizedEquation();
testPBRShaderContracts();
testPhongSpecularLobeNormalization();
console.log("Physical lighting contract tests passed");
