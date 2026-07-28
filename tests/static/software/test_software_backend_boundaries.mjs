import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

function read(relativePath) {
	return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

const backendSource = read("../../../src/backends/software/SoftwareBackend.ts");
assert.equal(
	/from "\.\/Rasterizer"|from "\.\/passes\//.test(backendSource),
	false,
	"SoftwareBackend must delegate concrete pass ownership to SoftwarePassExecutor"
);

const shaderSources = [
	read("../../../src/shaders/software/PBRStrategy.ts"),
	read("../../../src/shaders/software/BlinnPhongStrategy.ts"),
];
for (const source of shaderSources) {
	assert.equal(
		source.includes("backends/software/LightEvaluator"),
		false,
		"Software shader strategies must depend on the lighting runtime layer"
	);
}

const volumetricSource = read(
	"../../../src/postprocess/passes/VolumetricLightingPass.ts"
);
assert.equal(
	volumetricSource.includes("backends/software/passes/SoftwareShadowPass"),
	false,
	"Volumetric lighting must receive shadow sampling through its backend-neutral context"
);

const publicIndex = read("../../../src/index.ts");
assert.equal(
	/public\s+\{\s*Rasterizer\s*\}|export\s+\{\s*Rasterizer\s*\}/.test(publicIndex), false);

const rasterizerSource = read("../../../src/backends/software/Rasterizer.ts");
const drawTriangleDeclaration = rasterizerSource.match(
	/public drawTriangle\([\s\S]*?\n\t\): void \{/
);
assert.ok(drawTriangleDeclaration, "Rasterizer.drawTriangle declaration must exist");
assert.equal(
	drawTriangleDeclaration[0].includes("decalPackets"),
	false,
	"Rasterizer.drawTriangle must consume a prepared fragment program"
);

console.log("Software backend boundary tests passed");
