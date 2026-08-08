import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

function readSource(fileName) {
	return readFileSync(
		resolve(process.cwd(), "src/backends/webgpu", fileName),
		"utf8",
	);
}

const runtimeSource = readSource("WebGPUShadowRuntime.ts");
const atlasSource = readSource("WebGPUAtlasShadowTechnique.ts");
const pagedSource = readSource("WebGPUPagedShadowTechnique.ts");
const casterSource = readSource("WebGPUShadowCasterRenderer.ts");

assert.equal(
	existsSync(resolve(process.cwd(), "src/backends/webgpu/WebGPUShadowPass.ts")),
	false,
);
assert.equal(
	existsSync(
		resolve(process.cwd(), "src/backends/webgpu/WebGPUPagedShadowRuntime.ts"),
	),
	false,
);

assert.match(
	runtimeSource,
	/private readonly _casterRenderer: WebGPUShadowCasterRenderer/,
);
assert.match(
	runtimeSource,
	/private readonly _atlasTechnique: WebGPUAtlasShadowTechnique/,
);
assert.match(
	runtimeSource,
	/private readonly _pagedTechnique: WebGPUPagedShadowTechnique/,
);
assert.doesNotMatch(runtimeSource, /new WebGPUShadowPass/);
assert.doesNotMatch(runtimeSource, /public readonly pagedTechnique/);
assert.doesNotMatch(runtimeSource, /public readonly atlasAllocator/);

assert.match(atlasSource, /private readonly _allocator: WebGPUShadowAtlasAllocator/);
assert.match(atlasSource, /private readonly _casterRenderer: WebGPUShadowCasterRenderer/);
assert.doesNotMatch(atlasSource, /WebGPUPagedShadowTechnique/);

assert.match(pagedSource, /private _casterRenderer: WebGPUShadowCasterRenderer/);
assert.doesNotMatch(pagedSource, /WebGPUShadowPass/);
assert.doesNotMatch(pagedSource, /private _shadowPass/);
assert.doesNotMatch(pagedSource, /WebGPUAtlasShadowTechnique/);

assert.match(casterSource, /export class WebGPUShadowCasterRenderer/);
assert.doesNotMatch(casterSource, /private _shadowAtlases/);

console.log("WebGPU shadow runtime structure tests passed");
