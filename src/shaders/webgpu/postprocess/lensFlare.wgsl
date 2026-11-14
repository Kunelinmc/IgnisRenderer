// Scalar-only uniform layout: fifteen floats, matching LensFlarePass.
struct LensFlareParams {
	threshold: f32,
	softKnee: f32,
	intensity: f32,
	ghostIntensity: f32,
	ghostCount: f32,
	ghostSpacing: f32,
	haloIntensity: f32,
	haloRadius: f32,
	haloWidth: f32,
	chromaticAberration: f32,
	blurRadius: f32,
	edgeFade: f32,
	aspectRatio: f32,
	ghostRadius: f32,
	maxGhostScale: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: LensFlareParams;
@group(0) @binding(3) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var flareTex: texture_2d<f32>;

fn flareThresholdWeight(luma: f32, threshold: f32, softKnee: f32) -> f32 {
	let knee = max(softKnee, 0.0);
	let soft = clamp(luma - threshold + knee, 0.0, 2.0 * knee);
	let contribution = max(luma - threshold, soft * soft / max(4.0 * knee, 0.00001));
	let weight = clamp(contribution / max(luma, 0.00001), 0.0, 1.0);
	return select(0.0, weight, luma > 0.00001);
}

fn flareExtract(color: vec4<f32>, threshold: f32, softKnee: f32) -> vec4<f32> {
	let alpha = clamp(color.w, 0.0, 1.0);
	let rgb = vec3<f32>(max(color.x, 0.0), max(color.y, 0.0), max(color.z, 0.0));
	let straight = rgb / max(alpha, 0.00001);
	let luma = dot(straight, vec3<f32>(0.2126, 0.7152, 0.0722));
	let weight = flareThresholdWeight(luma, threshold, softKnee);
	return select(vec4<f32>(0.0), vec4<f32>(rgb, alpha) * weight, alpha > 0.0);
}

fn flareUvValid(uv: vec2<f32>) -> bool {
	return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn flareEdgeWeight(uv: vec2<f32>, width: f32) -> f32 {
	let distance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
	let t = clamp(distance / max(width, 0.00001), 0.0, 1.0);
	return select(t * t * (3.0 - 2.0 * t), 1.0, width <= 0.0);
}

fn flareGhostSourceScale(spacing: f32, index: f32, maxScale: f32) -> f32 {
	let nominal = -1.0 + spacing * index;
	// Bound magnification smoothly without collapsing several ghosts to one scale.
	let direction = select(-1.0, 1.0, nominal > 0.0);
	return nominal + direction / maxScale;
}

fn flareGhostUv(uv: vec2<f32>, spacing: f32, index: f32, maxScale: f32) -> vec2<f32> {
	let sourceScale = flareGhostSourceScale(spacing, index, maxScale);
	return vec2<f32>(0.5) + (uv - vec2<f32>(0.5)) * sourceScale;
}

fn flareGhostWeight(sourceScale: f32, index: f32) -> f32 {
	let magnification = 1.0 / abs(sourceScale);
	let area = max(1.0, magnification * magnification);
	return 1.0 / (area * (1.0 + 0.35 * index));
}

fn flareApertureSteps(radiusPixels: f32) -> f32 {
	return max(1.0, ceil(radiusPixels));
}

fn flareApertureAxis(index: f32) -> vec2<f32> {
	return select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), index < 0.5);
}

fn flareApertureOffset(axis: vec2<f32>, radiusPixels: vec2<f32>, index: f32, steps: f32) -> vec2<f32> {
	return axis * radiusPixels * (index / steps);
}

fn flareApertureSigma(radiusPixels: f32) -> f32 {
	return max(0.5, radiusPixels * 0.125);
}

fn flareGaussianWeight(distance: f32, sigma: f32) -> f32 {
	return exp(-0.5 * distance * distance / (sigma * sigma));
}

fn flareAperturePoint(q: f32, r: f32) -> vec2<f32> {
	return vec2<f32>(q + 0.5 * r, 0.8660254 * r) / 4.0;
}

fn flareApertureWeight(q: f32, r: f32) -> f32 {
	let point = flareAperturePoint(q, r);
	let boundary = max(max(abs(q), abs(r)), abs(q + r));
	let edgeWeight = select(0.5, 0.3333333, dot(point, point) > 0.99);
	return select(1.0, edgeWeight, boundary > 3.5);
}

fn flareHaloOffset(uv: vec2<f32>, aspect: f32, radius: f32) -> vec2<f32> {
	let scale = vec2<f32>(aspect, 1.0);
	let centered = (vec2<f32>(0.5) - uv) * scale;
	// The tiny third component keeps the exact optical center finite.
	let direction = normalize(vec3<f32>(centered, 0.000001)).xy;
	return direction * radius / scale;
}

fn flareHaloWeight(targetUv: vec2<f32>, sourceUv: vec2<f32>, aspect: f32) -> f32 {
	let scale = vec2<f32>(aspect, 1.0);
	let targetDelta = (targetUv - vec2<f32>(0.5)) * scale;
	let sourceDelta = (sourceUv - vec2<f32>(0.5)) * scale;
	// Compensate the radial warp's area enlargement, especially near the optical center.
	return min(1.0, sqrt(dot(sourceDelta, sourceDelta) /
		max(dot(targetDelta, targetDelta), 0.00000001)));
}

fn flareComposite(scene: vec4<f32>, flare: vec4<f32>, intensity: f32) -> vec4<f32> {
	let alpha = clamp(scene.w, 0.0, 1.0);
	let coverage = clamp(flare.w * intensity, 0.0, 1.0);
	let outputAlpha = alpha + coverage * (1.0 - alpha);
	let rgb = scene.xyz + flare.xyz * intensity;
	return select(vec4<f32>(0.0), vec4<f32>(rgb, outputAlpha), outputAlpha > 0.0);
}

fn finiteColor(color: vec4<f32>) -> vec4<f32> {
	if (color.a <= 0.0) { return vec4<f32>(0.0); }
	return vec4<f32>(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(65504.0)),
		clamp(color.a, 0.0, 1.0));
}

fn sampleImage(uv: vec2<f32>) -> vec4<f32> {
	// Do not wrap or clamp an out-of-image highlight onto the opposite edge.
	if (!flareUvValid(uv)) { return vec4<f32>(0.0); }
	return textureSampleLevel(srcTex, linearSampler, uv, 0.0);
}

fn sampleChromatic(uv: vec2<f32>, offset: vec2<f32>) -> vec4<f32> {
	let red = sampleImage(uv + offset);
	let green = sampleImage(uv);
	let blue = sampleImage(uv - offset);
	// Coverage spans all three displaced channel footprints.
	return vec4<f32>(red.r, green.g, blue.b, max(red.a, max(green.a, blue.a)));
}

fn sampleHaloHighlight(uv: vec2<f32>) -> vec4<f32> {
	if (!flareUvValid(uv)) { return vec4<f32>(0.0); }
	return textureSampleLevel(flareTex, linearSampler, uv, 0.0);
}

fn sampleHaloChromatic(uv: vec2<f32>, offset: vec2<f32>) -> vec4<f32> {
	let red = sampleHaloHighlight(uv + offset);
	let green = sampleHaloHighlight(uv);
	let blue = sampleHaloHighlight(uv - offset);
	return vec4<f32>(red.r, green.g, blue.b, max(red.a, max(green.a, blue.a)));
}

@compute @workgroup_size(8, 8, 1)
fn csExtract(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	var result = vec4<f32>(0.0);
	// Threshold individual taps so tiny HDR sources survive averaging.
	for (var y = 0; y < 4; y++) {
		for (var x = 0; x < 4; x++) {
			let uv = (vec2<f32>(gid.xy) +
				(vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) / 4.0) / vec2<f32>(size);
			let color = textureSampleLevel(srcTex, linearSampler, uv, 0.0);
			result += flareExtract(color, params.threshold, params.softKnee) *
				flareEdgeWeight(uv, params.edgeFade);
		}
	}
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(result / 16.0));
}

fn aperturePrefilter(gid: vec3<u32>, axis: vec2<f32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let texels = vec2<f32>(size);
	let radiusPixels = params.ghostRadius * texels / vec2<f32>(params.aspectRatio, 1.0);
	let sigma = flareApertureSigma(dot(radiusPixels, axis));
	let extent = 3.0 * sigma;
	let steps = i32(flareApertureSteps(extent));
	var result = vec4<f32>(0.0);
	var weightSum = 0.0;
	// Match the prefilter footprint to the hexagonal tap spacing to avoid a dotted lattice.
	for (var i = -steps; i <= steps; i++) {
		let offset = flareApertureOffset(axis, vec2<f32>(extent), f32(i), f32(steps));
		let weight = flareGaussianWeight(extent * f32(i) / f32(steps), sigma);
		result += sampleImage(uv + offset / texels) * weight;
		weightSum += weight;
	}
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(result / max(weightSum, 0.00001)));
}

@compute @workgroup_size(8, 8, 1)
fn csAperturePrefilterH(@builtin(global_invocation_id) gid: vec3<u32>) {
	aperturePrefilter(gid, flareApertureAxis(0.0));
}

@compute @workgroup_size(8, 8, 1)
fn csAperturePrefilterV(@builtin(global_invocation_id) gid: vec3<u32>) {
	aperturePrefilter(gid, flareApertureAxis(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csAperture(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let radius = params.ghostRadius / vec2<f32>(params.aspectRatio, 1.0);
	var result = vec4<f32>(0.0);
	var weightSum = 0.0;
	for (var r = -4; r <= 4; r++) {
		for (var q = -4; q <= 4; q++) {
			if (abs(q + r) > 4) { continue; }
			let point = flareAperturePoint(f32(q), f32(r));
			let weight = flareApertureWeight(f32(q), f32(r));
			result += sampleImage(uv + point * radius) * weight;
			weightSum += weight;
		}
	}
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(result / weightSum));
}

@compute @workgroup_size(8, 8, 1)
fn csFeatures(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let separation = flareHaloOffset(uv, params.aspectRatio, params.chromaticAberration);
	var result = vec4<f32>(0.0);
	if (params.ghostIntensity > 0.0) {
		for (var i = 0; i < i32(params.ghostCount); i++) {
			let sourceScale = flareGhostSourceScale(params.ghostSpacing, f32(i), params.maxGhostScale);
			let sourceUv = flareGhostUv(uv, params.ghostSpacing, f32(i), params.maxGhostScale);
			let weight = flareGhostWeight(sourceScale, f32(i));
			let tint = mix(vec3<f32>(0.9, 0.96, 1.0), vec3<f32>(1.0, 0.93, 0.83),
				f32(i) / max(params.ghostCount - 1.0, 1.0));
			// Compensate the source-space offset to keep dispersion fixed on screen.
			let ghost = sampleChromatic(sourceUv, separation * abs(sourceScale));
			result += vec4<f32>(ghost.rgb * tint, ghost.a) *
				(params.ghostIntensity * weight / params.ghostCount);
		}
	}
	if (params.haloIntensity > 0.0) {
		let flipped = vec2<f32>(1.0) - uv;
		let steps = i32(flareApertureSteps(params.haloWidth * f32(textureDimensions(flareTex).y)));
		var halo = vec4<f32>(0.0);
		var weightSum = 0.0;
		for (var i = -steps; i <= steps; i++) {
			let t = f32(i) / f32(steps);
			let radius = max(params.haloRadius + t * params.haloWidth, 0.0);
			let offset = flareHaloOffset(flipped, params.aspectRatio, radius);
			let weight = (1.0 - t * t) * (1.0 - t * t);
			let areaWeight = flareHaloWeight(flipped, flipped + offset, params.aspectRatio);
			halo += sampleHaloChromatic(flipped + offset, separation) * (weight * areaWeight);
			weightSum += weight;
		}
		result += halo * (params.haloIntensity / max(weightSum, 0.00001));
	}
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(result));
}

fn blur(gid: vec3<u32>, axis: vec2<f32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let step = axis * params.blurRadius / vec2<f32>(textureDimensions(srcTex));
	var color = sampleImage(uv) * 0.375;
	color += (sampleImage(uv - step) + sampleImage(uv + step)) * 0.25;
	color += (sampleImage(uv - step * 2.0) + sampleImage(uv + step * 2.0)) * 0.0625;
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(color));
}

@compute @workgroup_size(8, 8, 1)
fn csBlurH(@builtin(global_invocation_id) gid: vec3<u32>) {
	blur(gid, vec2<f32>(1.0, 0.0));
}

@compute @workgroup_size(8, 8, 1)
fn csBlurV(@builtin(global_invocation_id) gid: vec3<u32>) {
	blur(gid, vec2<f32>(0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn csComposite(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(dstTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
	let scene = textureLoad(srcTex, vec2<i32>(gid.xy), 0);
	let flare = textureSampleLevel(flareTex, linearSampler, uv, 0.0);
	textureStore(dstTex, vec2<i32>(gid.xy), finiteColor(flareComposite(scene, flare, params.intensity)));
}
