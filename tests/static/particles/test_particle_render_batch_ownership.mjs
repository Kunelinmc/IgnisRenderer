import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const sourceRoot = join(import.meta.dir, "../../../src");
const renderBatchSource = readFileSync(
	join(sourceRoot, "particles/ParticleRenderBatch.ts"),
	"utf8",
);
const particleIndexSource = readFileSync(
	join(sourceRoot, "particles/index.ts"),
	"utf8",
);
const pipelineTypesSource = readFileSync(
	join(sourceRoot, "pipeline/types.ts"),
	"utf8",
);

for (const interfaceName of [
	"ParticleUVRect",
	"ParticleRenderItem",
	"ParticleRenderBatch",
	"ParticleMeshRenderItem",
	"ParticleMeshRenderBatch",
]) {
	assert.match(
		renderBatchSource,
		new RegExp(`export interface ${interfaceName}\\b`),
	);
	assert.doesNotMatch(
		pipelineTypesSource,
		new RegExp(`export interface ${interfaceName}\\b`),
	);
}

assert.match(
	pipelineTypesSource,
	/from "\.\.\/particles\/ParticleRenderBatch"/,
);
assert.doesNotMatch(particleIndexSource, /ParticleRenderBatch/);

console.log("Particle render-batch ownership tests passed");
