import assert from "node:assert/strict";
import { resolveFeatureState } from "../../../src/pipeline/FeatureResolver.ts";
import { WebGPUBackend } from "../../../src/backends/webgpu/WebGPUBackend.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";
import { attachBackend } from "../../helpers/TestRenderBackend.mjs";

function testEnableOITNegotiationAcrossBackends() {
	const webgpu = attachBackend(new WebGPUBackend());
	const webgl = attachBackend(new WebGLBackend());
	const software = attachBackend(new SoftwareBackend());

	const request = {
		enableLighting: true,
		enableGamma: true,
		enableOIT: true,
	};

	const webgpuResolved = resolveFeatureState(
		request,
		webgpu.profile.capabilities,
		webgpu.profile.id
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
		webgl.profile.capabilities,
		webgl.profile.id
	);
	assert.equal(webglResolved.enableOIT, true);
	assert.equal(
		webglResolved.warnings.some((warning) =>
			warning.key === "webgl-feature-oit"
		),
		false
	);

	const softwareResolved = resolveFeatureState(
		request,
		software.profile.capabilities,
		software.profile.id
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
	const webgl = attachBackend(new WebGLBackend());
	const resolved = resolveFeatureState(
		{
			enableLighting: true,
			enableGamma: true,
			enableOIT: false,
		},
		webgl.profile.capabilities,
		webgl.profile.id
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
