const FXAA_QUALITY = array<f32, 11>(
	1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0
);

struct Params {
	invSize: vec2<f32>,
	edgeThresholdMin: f32,
	edgeThresholdMultiplier: f32,
	subpixQuality: f32,
	_pad0: f32,
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba16float, write>;

fn perceptualLuma(rgb: vec3<f32>) -> f32 {
	let linearLuma = max(
		dot(max(rgb, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722)),
		0.0
	);
	return sqrt(linearLuma);
}

fn loadColorClamped(coord: vec2<i32>) -> vec4<f32> {
	let size = vec2<i32>(textureDimensions(srcTex));
	let clamped = clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
	return textureLoad(srcTex, clamped, 0);
}

fn loadLuma(coord: vec2<i32>) -> f32 {
	return perceptualLuma(loadColorClamped(coord).rgb);
}

fn sampleColor(pixelPos: vec2<f32>) -> vec4<f32> {
	let uv = (pixelPos + vec2<f32>(0.5, 0.5)) * params.invSize;
	return textureSampleLevel(srcTex, linearSampler, uv, 0.0);
}

fn sampleLuma(pixelPos: vec2<f32>) -> f32 {
	return perceptualLuma(sampleColor(pixelPos).rgb);
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outTex);
	if (gid.x >= size.x || gid.y >= size.y) { return; }

	let coord = vec2<i32>(gid.xy);
	let centerPos = vec2<f32>(coord);
	let centerColor = textureLoad(srcTex, coord, 0);
	let lumaCenter = perceptualLuma(centerColor.rgb);
	let lumaNorth = loadLuma(coord + vec2<i32>(0, -1));
	let lumaSouth = loadLuma(coord + vec2<i32>(0, 1));
	let lumaEast = loadLuma(coord + vec2<i32>(1, 0));
	let lumaWest = loadLuma(coord + vec2<i32>(-1, 0));

	let lumaMin = min(
		lumaCenter,
		min(min(lumaNorth, lumaSouth), min(lumaEast, lumaWest))
	);
	let lumaMax = max(
		lumaCenter,
		max(max(lumaNorth, lumaSouth), max(lumaEast, lumaWest))
	);
	let lumaRange = lumaMax - lumaMin;

	if (
		lumaRange <
		max(params.edgeThresholdMin, lumaMax * params.edgeThresholdMultiplier)
	) {
		textureStore(outTex, coord, centerColor);
		return;
	}

	let lumaNorthWest = loadLuma(coord + vec2<i32>(-1, -1));
	let lumaNorthEast = loadLuma(coord + vec2<i32>(1, -1));
	let lumaSouthWest = loadLuma(coord + vec2<i32>(-1, 1));
	let lumaSouthEast = loadLuma(coord + vec2<i32>(1, 1));

	var filteredLuma = 2.0 * (lumaNorth + lumaSouth + lumaEast + lumaWest);
	filteredLuma += lumaNorthEast + lumaNorthWest + lumaSouthEast + lumaSouthWest;
	filteredLuma /= 12.0;

	let subpixOffset1 = abs(filteredLuma - lumaCenter);
	let subpixOffset2 = clamp(subpixOffset1 / max(lumaRange, 1e-4), 0.0, 1.0);
	let subpixOffset3 =
		(-2.0 * subpixOffset2 + 3.0) * subpixOffset2 * subpixOffset2;
	let subpixOffset =
		subpixOffset3 * subpixOffset3 * params.subpixQuality;

	let edgeHorz =
		abs(-2.0 * lumaWest + lumaNorthWest + lumaSouthWest) +
		abs(-2.0 * lumaCenter + lumaNorth + lumaSouth) * 2.0 +
		abs(-2.0 * lumaEast + lumaNorthEast + lumaSouthEast);
	let edgeVert =
		abs(-2.0 * lumaNorth + lumaNorthWest + lumaNorthEast) +
		abs(-2.0 * lumaCenter + lumaWest + lumaEast) * 2.0 +
		abs(-2.0 * lumaSouth + lumaSouthWest + lumaSouthEast);
	let isHorz = edgeHorz >= edgeVert;

	let lumaA = select(lumaWest, lumaNorth, isHorz);
	let lumaB = select(lumaEast, lumaSouth, isHorz);
	let gradientA = abs(lumaA - lumaCenter);
	let gradientB = abs(lumaB - lumaCenter);
	let isASteeper = gradientA >= gradientB;
	let gradientScaled = 0.25 * max(gradientA, gradientB);
	let stepSign = select(1.0, -1.0, isASteeper);
	let edgeLuma = select(
		(lumaB + lumaCenter) * 0.5,
		(lumaA + lumaCenter) * 0.5,
		isASteeper
	);

	var posN = centerPos;
	if (isHorz) {
		posN.y += stepSign * 0.5;
	} else {
		posN.x += stepSign * 0.5;
	}
	var posP = posN;
	let axis = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), isHorz);

	var doneN = false;
	var doneP = false;
	var lumaEndN = 0.0;
	var lumaEndP = 0.0;
	for (var i: i32 = 0; i < 11; i = i + 1) {
		if (!doneN) {
			lumaEndN = sampleLuma(posN);
			doneN = abs(lumaEndN - edgeLuma) >= gradientScaled;
		}
		if (!doneP) {
			lumaEndP = sampleLuma(posP);
			doneP = abs(lumaEndP - edgeLuma) >= gradientScaled;
		}
		if (doneN && doneP) {
			break;
		}
		if (!doneN) {
			posN -= axis * FXAA_QUALITY[i];
		}
		if (!doneP) {
			posP += axis * FXAA_QUALITY[i];
		}
	}

	let distN = select(centerPos.y - posN.y, centerPos.x - posN.x, isHorz);
	let distP = select(posP.y - centerPos.y, posP.x - centerPos.x, isHorz);
	let isNCloser = distN < distP;
	let distMin = min(distN, distP);
	let lumaEndMin = select(lumaEndP, lumaEndN, isNCloser);
	let isCenterBright = (lumaCenter - edgeLuma) >= 0.0;
	let isEndBright = (lumaEndMin - edgeLuma) >= 0.0;
	let reachedProperly = isEndBright != isCenterBright;

	var edgeOffset = -distMin / max(distN + distP, 1e-4) + 0.5;
	if (!reachedProperly) {
		edgeOffset = 0.0;
	}
	let pixelOffset = max(subpixOffset, edgeOffset);

	var finalPos = centerPos;
	if (isHorz) {
		finalPos.y += stepSign * pixelOffset;
	} else {
		finalPos.x += stepSign * pixelOffset;
	}

	textureStore(outTex, coord, sampleColor(finalPos));
}
