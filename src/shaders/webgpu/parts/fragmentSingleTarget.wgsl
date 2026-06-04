fn resolveOITWeight(alpha: f32, linearDepth: f32) -> f32 {
	let clampedAlpha = clamp(alpha, 0.0, 1.0);
	let normalizedDepth = clamp(linearDepth / 400.0, 0.0, 1.0);
	let depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	let alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	let weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

fn buildSceneOITOutput(sceneColor: vec4<f32>, linearDepth: f32) -> SceneFragmentOITOutput {
	let alpha = clamp(sceneColor.a, 0.0, 1.0);
	let weight = resolveOITWeight(alpha, linearDepth);
	var output: SceneFragmentOITOutput;
	output.accum = vec4<f32>(sceneColor.rgb * alpha, alpha) * weight;
	output.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
	return output;
}

@fragment
fn fsMain(input: VertexOutput) -> SceneFragmentOutput {
	return shadeScene(input);
}

@fragment
fn fsMainSingle(input: VertexOutput) -> @location(0) vec4<f32> {
	return shadeScene(input).sceneColor;
}

@fragment
fn fsMainOIT(input: VertexOutput) -> SceneFragmentOITOutput {
	let shaded = shadeScene(input);
	return buildSceneOITOutput(shaded.sceneColor, shaded.gMotionDepth.z);
}

@fragment
fn fsMainDepthMask(input: VertexOutput) {
	let alphaModeMask = model.materialFlags.y > 0.5;
	if (!alphaModeMask) {
		return;
	}
	let isWireframe = model.materialFlags.w > 0.5;
	var baseSample = vec4<f32>(1.0);
	if (!isWireframe) {
		baseSample = sampleColorTexture(
			baseColorTexture,
			baseColorSampler,
			TEX_BASE_COLOR,
			input.uv0,
			input.uv1,
			input.uv2,
			input.uv3
		);
	}
	let alpha = clamp(model.baseColorFactor.a * baseSample.a, 0.0, 1.0);
	if (alpha < model.surfaceParams0.w) {
		discard;
	}
}
