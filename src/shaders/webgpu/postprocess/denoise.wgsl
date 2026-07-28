struct DenoiseParams {
	invSignalSize: vec2<f32>,
	radius: f32,
	stepWidth: f32,
	depthPhi: f32,
	normalPhi: f32,
	valuePhi: f32,
	confidenceFloor: f32,
	signalMode: f32,
	qualityMode: f32,
	_pad0: vec2<f32>,
	_pad1: vec4<f32>,
}

@group(0) @binding(0) var signalSource: texture_2d<f32>;
@group(0) @binding(1) var normalTexture: texture_2d<f32>;
@group(0) @binding(2) var depthTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var<uniform> params: DenoiseParams;
@group(0) @binding(5)
var signalOut: texture_storage_2d<rgba16float, write>;

const WORKGROUP_SIZE = 8u;
const TILE_EXTENT = 24u;
const TILE_COUNT = 192u;
const TILE_HALO = 8i;
const MAX_FINITE_VALUE = 3.4e38;
const MAX_SIGNAL_VALUE = 65504.0;
const LUMINANCE = vec3<f32>(0.2126, 0.7152, 0.0722);

var<workgroup> tileSignal: array<vec4<f32>, TILE_COUNT>;
var<workgroup> tileDepth: array<f32, TILE_COUNT>;
var<workgroup> tileNormal: array<vec4<f32>, TILE_COUNT>;

fn signNotZero2(value: vec2<f32>) -> vec2<f32> {
	return vec2<f32>(
		select(-1.0, 1.0, value.x >= 0.0),
		select(-1.0, 1.0, value.y >= 0.0)
	);
}

fn octahedralWrap(value: vec2<f32>) -> vec2<f32> {
	return (vec2<f32>(1.0) - abs(value.yx)) * signNotZero2(value);
}

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let oct = encoded * 2.0 - vec2<f32>(1.0);
	var normal = vec3<f32>(
		oct.x,
		oct.y,
		1.0 - abs(oct.x) - abs(oct.y)
	);
	if (normal.z < 0.0) {
		normal = vec3<f32>(octahedralWrap(normal.xy), normal.z);
	}
	return normalize(normal);
}

fn clampSignalCoord(coord: vec2<i32>, size: vec2<u32>) -> vec2<i32> {
	return clamp(
		coord,
		vec2<i32>(0),
		vec2<i32>(size) - vec2<i32>(1)
	);
}

fn loadTileValue(index: u32, coord: vec2<i32>, size: vec2<u32>) {
	let safeCoord = clampSignalCoord(coord, size);
	let uv = (vec2<f32>(safeCoord) + vec2<f32>(0.5)) *
		params.invSignalSize;
	tileSignal[index] = textureLoad(signalSource, safeCoord, 0);
	tileDepth[index] = textureSampleLevel(
		depthTexture,
		linearSampler,
		uv,
		0.0
	).z;
	tileNormal[index] = vec4<f32>(
		decodeNormal(
			textureSampleLevel(
				normalTexture,
				linearSampler,
				uv,
				0.0
			).xy
		),
		0.0
	);
}

fn loadHorizontalTile(
	localIndex: u32,
	workgroupId: vec3<u32>,
	size: vec2<u32>
) {
	let origin = vec2<i32>(workgroupId.xy * WORKGROUP_SIZE);
	var index = localIndex;
	for (; index < TILE_COUNT; index += WORKGROUP_SIZE * WORKGROUP_SIZE) {
		let tileX = i32(index % TILE_EXTENT);
		let tileY = i32(index / TILE_EXTENT);
		loadTileValue(
			index,
			origin + vec2<i32>(tileX - TILE_HALO, tileY),
			size
		);
	}
}

fn loadVerticalTile(
	localIndex: u32,
	workgroupId: vec3<u32>,
	size: vec2<u32>
) {
	let origin = vec2<i32>(workgroupId.xy * WORKGROUP_SIZE);
	var index = localIndex;
	for (; index < TILE_COUNT; index += WORKGROUP_SIZE * WORKGROUP_SIZE) {
		let tileX = i32(index % WORKGROUP_SIZE);
		let tileY = i32(index / WORKGROUP_SIZE);
		loadTileValue(
			index,
			origin + vec2<i32>(tileX, tileY - TILE_HALO),
			size
		);
	}
}

fn sanitizeSignal(value: vec4<f32>) -> vec4<f32> {
	if (params.signalMode > 0.5) {
		let nonNanScalar = select(value.x, 0.0, value.x != value.x);
		let scalar = clamp(nonNanScalar, 0.0, 1.0);
		return vec4<f32>(vec3<f32>(scalar), 1.0);
	}
	let nonNanColor = select(
		value.rgb,
		vec3<f32>(0.0),
		value.rgb != value.rgb
	);
	let nonNanConfidence = select(value.a, 0.0, value.a != value.a);
	return vec4<f32>(
		clamp(nonNanColor, vec3<f32>(0.0), vec3<f32>(MAX_SIGNAL_VALUE)),
		clamp(nonNanConfidence, 0.0, 1.0)
	);
}

fn validDepth(value: f32) -> bool {
	return value == value &&
		value > 0.0 &&
		abs(value) <= MAX_FINITE_VALUE;
}

fn safeNonNegative(value: f32) -> f32 {
	let nonNanValue = select(value, 0.0, value != value);
	return clamp(nonNanValue, 0.0, MAX_SIGNAL_VALUE);
}

fn safePositiveInteger(value: f32, maximum: i32) -> i32 {
	let nonNanValue = select(value, 1.0, value != value);
	return i32(clamp(nonNanValue, 1.0, f32(maximum)));
}

fn signalValue(value: vec4<f32>) -> f32 {
	return select(
		max(dot(max(value.rgb, vec3<f32>(0.0)), LUMINANCE), 0.0),
		clamp(value.x, 0.0, 1.0),
		params.signalMode > 0.5
	);
}

fn signalValueDistance(centerValue: f32, sampleValue: f32) -> f32 {
	if (params.signalMode > 0.5) {
		return abs(sampleValue - centerValue);
	}
	return abs(log(1.0 + sampleValue) - log(1.0 + centerValue));
}

fn spatialWeight(offset: i32, radius: i32) -> f32 {
	if (params.qualityMode > 0.5) {
		let distance = abs(offset);
		if (distance == 0) {
			return 6.0;
		}
		if (distance == 1) {
			return 4.0;
		}
		return 1.0;
	}
	let normalizedOffset = f32(offset) / max(f32(radius), 1.0);
	return exp(-0.5 * normalizedOffset * normalizedOffset);
}

fn filterTile(centerIndex: u32, axisStride: i32) -> vec4<f32> {
	if (centerIndex >= TILE_COUNT) {
		return vec4<f32>(0.0);
	}
	let centerSignal = sanitizeSignal(tileSignal[centerIndex]);
	let centerDepth = tileDepth[centerIndex];
	if (!validDepth(centerDepth)) {
		return centerSignal;
	}
	let centerNormal = tileNormal[centerIndex].xyz;
	let centerValue = signalValue(centerSignal);
	let radius = safePositiveInteger(params.radius, 4);
	let requestedStepWidth = safePositiveInteger(params.stepWidth, 4);
	// Keep the footprint within the preloaded halo even for malformed uniforms.
	let stepWidth = min(
		requestedStepWidth,
		max(TILE_HALO / radius, 1)
	);
	let depthPhi = safeNonNegative(params.depthPhi);
	let normalPhi = safeNonNegative(params.normalPhi);
	let valuePhi = safeNonNegative(params.valuePhi);
	let confidenceFloor = clamp(
		safeNonNegative(params.confidenceFloor),
		0.0,
		1.0
	);
	var colorSum = vec3<f32>(0.0);
	var scalarSum = 0.0;
	var confidenceSum = 0.0;
	var signalWeightSum = 0.0;
	var baseWeightSum = 0.0;

	for (var offset = -4; offset <= 4; offset++) {
		if (abs(offset) > radius) {
			continue;
		}
		let tileOffset = offset * stepWidth * axisStride;
		let sampleIndexSigned = i32(centerIndex) + tileOffset;
		if (sampleIndexSigned < 0 ||
			sampleIndexSigned >= i32(TILE_COUNT)) {
			continue;
		}
		let sampleIndex = u32(sampleIndexSigned);
		let sampleDepth = tileDepth[sampleIndex];
		if (!validDepth(sampleDepth)) {
			continue;
		}
		let sampleSignal = sanitizeSignal(tileSignal[sampleIndex]);
		let sampleNormal = tileNormal[sampleIndex].xyz;
		let relativeDepth =
			abs(sampleDepth - centerDepth) /
			max(max(sampleDepth, centerDepth), 1e-4);
		let depthWeight = select(
			1.0,
			exp(-relativeDepth * depthPhi),
			depthPhi > 0.0
		);
		let normalWeight = select(
			1.0,
			pow(
				max(dot(centerNormal, sampleNormal), 0.0),
				max(normalPhi, 0.001)
			),
			normalPhi > 0.0
		);
		let sampleValue = signalValue(sampleSignal);
		let valueDistance = signalValueDistance(centerValue, sampleValue);
		let rawValueWeight = select(
			1.0,
			exp(-valueDistance * valuePhi),
			valuePhi > 0.0
		);
		let valueTrust = select(
			min(centerSignal.a, sampleSignal.a),
			1.0,
			params.signalMode > 0.5
		);
		// Radiance differences only represent edges when both values are reliable.
		let valueWeight = mix(1.0, rawValueWeight, valueTrust);
		let baseWeight =
			spatialWeight(offset, radius) *
			depthWeight *
			normalWeight *
			valueWeight;
		baseWeightSum += baseWeight;

		if (params.signalMode > 0.5) {
			scalarSum += sampleSignal.x * baseWeight;
			signalWeightSum += baseWeight;
		} else {
			let reliability = mix(
				confidenceFloor,
				1.0,
				sampleSignal.a
			);
			let radianceWeight = baseWeight * reliability;
			colorSum += sampleSignal.rgb * radianceWeight;
			signalWeightSum += radianceWeight;
			confidenceSum += sampleSignal.a * baseWeight;
		}
	}

	if (signalWeightSum <= 1e-6 || baseWeightSum <= 1e-6) {
		return centerSignal;
	}
	if (params.signalMode > 0.5) {
		let scalar = clamp(scalarSum / signalWeightSum, 0.0, 1.0);
		return vec4<f32>(vec3<f32>(scalar), 1.0);
	}
	let neighborhoodConfidence = clamp(
		confidenceSum / baseWeightSum,
		0.0,
		1.0
	);
	let reliableSupport = clamp(
		signalWeightSum / baseWeightSum,
		0.0,
		1.0
	);
	// Low-reliability neighborhoods adjust confidence conservatively.
	return vec4<f32>(
		max(colorSum / signalWeightSum, vec3<f32>(0.0)),
		mix(centerSignal.a, neighborhoodConfidence, reliableSupport)
	);
}

@compute @workgroup_size(8, 8, 1)
fn csDenoiseHorizontal(
	@builtin(global_invocation_id) gid: vec3<u32>,
	@builtin(local_invocation_id) localId: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32,
	@builtin(workgroup_id) workgroupId: vec3<u32>
) {
	let size = textureDimensions(signalOut);
	loadHorizontalTile(localIndex, workgroupId, size);
	workgroupBarrier();
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let centerIndex =
		localId.y * TILE_EXTENT +
		localId.x +
		u32(TILE_HALO);
	textureStore(
		signalOut,
		vec2<i32>(gid.xy),
		filterTile(centerIndex, 1)
	);
}

@compute @workgroup_size(8, 8, 1)
fn csDenoiseVertical(
	@builtin(global_invocation_id) gid: vec3<u32>,
	@builtin(local_invocation_id) localId: vec3<u32>,
	@builtin(local_invocation_index) localIndex: u32,
	@builtin(workgroup_id) workgroupId: vec3<u32>
) {
	let size = textureDimensions(signalOut);
	loadVerticalTile(localIndex, workgroupId, size);
	workgroupBarrier();
	if (gid.x >= size.x || gid.y >= size.y) {
		return;
	}
	let centerIndex =
		(localId.y + u32(TILE_HALO)) * WORKGROUP_SIZE +
		localId.x;
	textureStore(
		signalOut,
		vec2<i32>(gid.xy),
		filterTile(centerIndex, i32(WORKGROUP_SIZE))
	);
}
