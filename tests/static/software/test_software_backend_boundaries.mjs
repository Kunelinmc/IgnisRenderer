import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

function read(relativePath) {
	return readFileSync(new URL(relativePath, import.meta.url), "utf8");
}

const backendSource = read("../../../src/backends/software/SoftwareBackend.ts");
assert.equal(
	/from "\.\/Rasterizer"|from "\.\/passes\//.test(backendSource),
	false,
	"SoftwareBackend must delegate concrete pass ownership to SoftwarePassExecutor"
);
assert.equal(
	/\.scene\.|\.attachments\.(pixels|depthBuffer)|Projector/.test(backendSource),
	false,
	"SoftwareBackend must remain a lifecycle-only facade"
);

const passSources = [
	"SoftwareMainPass.ts",
	"SoftwareParticlePass.ts",
	"SoftwareReflectionPass.ts",
	"SoftwareShadowPass.ts",
	"types.ts",
].map((name) => read(`../../../src/backends/software/passes/${name}`));
for (const source of passSources) {
	assert.equal(
		/\bFrameContext\b|\.transient\.(get|set)\(/.test(source),
		false,
		"Software render passes must consume SoftwarePassContext without transient access"
	);
}
const shadowCoordinatorOwnsExecutionKernel =
	/calculateShadowFactor|sampleParticleShadowVolumeTransmittance|drawDepthTriangle|drawTransmissionTriangle/.test(
		passSources[3]
	);
assert.equal(
	shadowCoordinatorOwnsExecutionKernel,
	false,
	"SoftwareShadowPass must coordinate work without owning raster or sampling kernels"
);

const softwareRuntimeSources = [
	backendSource,
	read("../../../src/backends/software/SoftwareFrameView.ts"),
	read("../../../src/backends/software/SoftwareFrameSession.ts"),
	read("../../../src/backends/software/SoftwareFrameServices.ts"),
	read("../../../src/backends/software/SoftwarePassExecutor.ts"),
	read("../../../src/backends/software/SoftwarePlanarReflectionRuntime.ts"),
	...passSources,
];
for (const source of softwareRuntimeSources) {
	assert.equal(
		/defineTransientKey|\.transient\.set\(/.test(source),
		false,
		"Software-owned runtime state must not be published through shared transients"
	);
}

const reflectionRuntimeSource = read(
	"../../../src/backends/software/SoftwarePlanarReflectionRuntime.ts"
);
for (const dependency of [
	"SoftwareReflectionPlanner",
	"SoftwareReflectionResources",
	"SoftwareReflectionRenderer",
	"SoftwareReflectionCompositor",
]) {
	assert.equal(
		reflectionRuntimeSource.includes(dependency),
		true,
		`Planar reflection runtime must delegate to ${dependency}`
	);
}

const servicesSource = read(
	"../../../src/backends/software/SoftwareFrameServices.ts"
);
for (const service of ["rasterizer", "material", "postProcess", "shadow", "reflection", "particles"]) {
	assert.match(
		servicesSource,
		new RegExp(`\\b${service}\\b`),
		`SoftwareFrameServices must expose ${service} explicitly`
	);
}

const shaderSources = [
	read("../../../src/shaders/software/PBRStrategy.ts"),
	read("../../../src/shaders/software/BlinnPhongStrategy.ts"),
];
for (const source of shaderSources) {
	assert.equal(
		source.includes("backends/software/"),
		false,
		"Software shader strategies must not depend on Software backend implementations"
	);
}

function collectTypeScriptFiles(directory) {
	const result = [];
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const path = resolve(directory, entry.name);
		if (entry.isDirectory()) result.push(...collectTypeScriptFiles(path));
		else if (entry.name.endsWith(".ts")) result.push(path);
	}
	return result;
}

function findSoftwareImportCycles() {
	const root = resolve(
		dirname(fileURLToPath(import.meta.url)),
		"../../../src/backends/software"
	);
	const files = collectTypeScriptFiles(root);
	const fileSet = new Set(files);
	const graph = new Map();
	for (const file of files) {
		const dependencies = [];
		for (const match of readFileSync(file, "utf8").matchAll(/from\s+["']([^"']+)["']/g)) {
			if (!match[1].startsWith(".")) continue;
			const dependency = resolve(dirname(file), `${match[1]}.ts`);
			if (fileSet.has(dependency)) dependencies.push(dependency);
		}
		graph.set(file, dependencies);
	}

	let nextIndex = 0;
	const stack = [];
	const onStack = new Set();
	const indices = new Map();
	const lowLinks = new Map();
	const cycles = [];
	const visit = (file) => {
		indices.set(file, nextIndex);
		lowLinks.set(file, nextIndex++);
		stack.push(file);
		onStack.add(file);
		for (const dependency of graph.get(file)) {
			if (!indices.has(dependency)) {
				visit(dependency);
				lowLinks.set(file, Math.min(lowLinks.get(file), lowLinks.get(dependency)));
			} else if (onStack.has(dependency)) {
				lowLinks.set(file, Math.min(lowLinks.get(file), indices.get(dependency)));
			}
		}
		if (lowLinks.get(file) !== indices.get(file)) return;
		const component = [];
		let member;
		do {
			member = stack.pop();
			onStack.delete(member);
			component.push(relative(root, member));
		} while (member !== file);
		if (component.length > 1) cycles.push(component);
	};
	for (const file of files) if (!indices.has(file)) visit(file);
	return cycles;
}

assert.deepEqual(
	findSoftwareImportCycles(),
	[],
	"Software backend import graph must not contain strongly connected components"
);

const volumetricSource = read(
	"../../../src/postprocess/passes/VolumetricLightingPass.ts"
);
assert.equal(
	volumetricSource.includes("backends/software/passes/SoftwareShadowPass"),
	false,
	"Volumetric lighting must not depend on the Software shadow pass"
);
assert.equal(
	volumetricSource.includes("SoftwareVolumetricLightingImplementation"),
	false,
	"Volumetric lighting must not define a Software implementation"
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
