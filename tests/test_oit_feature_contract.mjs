import assert from "node:assert/strict";
import { resolveFeatureState } from "../src/pipeline/FeatureResolver.ts";
import { WebGPUBackend } from "../src/renderers/WebGPUBackend.ts";
import { WebGLBackend } from "../src/renderers/WebGLBackend.ts";
import { SoftwareBackend } from "../src/renderers/SoftwareBackend.ts";

function testEnableOITNegotiationAcrossBackends() {
	const webgpu = new WebGPUBackend();
	const webgl = new WebGLBackend();
	const software = new SoftwareBackend();

	const request = {
		enableLighting: true,
		enableGamma: true,
		enableOIT: true,
	};

	const webgpuResolved = resolveFeatureState(
		request,
		webgpu.capabilities,
		webgpu.type
	);
	assert.equal(webgpuResolved.enableOIT, true);
	assert.equal(
		webgpuResolved.warnings.some((warning) =>
			warning.key === "webgpu-feature-oit"
		),
		false
	);

	const webglResolved = resolveFeatureState(
		request,
		webgl.capabilities,
		webgl.type
	);
	assert.equal(webglResolved.enableOIT, false);
	assert.equal(
		webglResolved.warnings.some((warning) =>
			warning.key === "webgl-feature-oit"
		),
		true
	);

	const softwareResolved = resolveFeatureState(
		request,
		software.capabilities,
		software.type
	);
	assert.equal(softwareResolved.enableOIT, false);
	assert.equal(
		softwareResolved.warnings.some((warning) =>
			warning.key === "software-feature-oit"
		),
		true
	);
}

function testEnableOITDisabledRequestHasNoWarning() {
	const webgl = new WebGLBackend();
	const resolved = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableOIT: false,
		},
		webgl.capabilities,
		webgl.type
	);
	assert.equal(resolved.enableOIT, false);
	assert.equal(
		resolved.warnings.some((warning) => warning.key === "webgl-feature-oit"),
		false
	);
}

function run() {
	testEnableOITNegotiationAcrossBackends();
	testEnableOITDisabledRequestHasNoWarning();
	console.log("OIT feature contract tests passed");
}

run();
