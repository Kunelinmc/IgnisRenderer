const EPSILON: f32 = 0.000001;

struct PagedShadowFeedbackParams {
	pageTableLength: u32,
	width: u32,
	height: u32,
	layoutCount: u32,
	frameId: u32,
	feedbackFlagCount: u32,
	_pad0: u32,
	_pad1: u32,
}

struct PagedShadowLayoutData {
	pageTableBase: u32,
	pageTableCascadeStride: u32,
	pageGridSize: u32,
	cascadeCount: u32,
	priorityBase: u32,
	feedbackMode: u32,
	_pad0: u32,
	_pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowFeedbackParams;
@group(0) @binding(1) var<storage, read_write> nextFeedbackFlags: array<u32>;
@group(0) @binding(2) var<storage, read> pageTable: array<u32>;
@group(0) @binding(3) var<storage, read> layouts: array<PagedShadowLayoutData>;
@group(0) @binding(4) var<storage, read> cascadeViewProjections: array<mat4x4<f32>>;
@group(0) @binding(5) var frameDepth: texture_depth_2d;
@group(0) @binding(6) var<storage, read> inverseViewProjection: array<mat4x4<f32>>;

fn reconstructWorldPosition(coord: vec2<u32>) -> vec4<f32> {
	let depth = textureLoad(frameDepth, vec2<i32>(coord), 0);
	if (depth >= 0.999999) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}
	let screen = (vec2<f32>(f32(coord.x), f32(coord.y)) + vec2<f32>(0.5)) /
		vec2<f32>(f32(max(params.width, 1u)), f32(max(params.height, 1u)));
	let ndc = vec3<f32>(
		screen.x * 2.0 - 1.0,
		1.0 - screen.y * 2.0,
		depth * 2.0 - 1.0
	);
	let world = inverseViewProjection[0] * vec4<f32>(ndc, 1.0);
	if (abs(world.w) <= EPSILON) {
		return vec4<f32>(0.0, 0.0, 0.0, 0.0);
	}
	return vec4<f32>(world.xyz / vec3<f32>(world.w), 1.0);
}

fn markFeedbackPage(worldPosition: vec3<f32>, layoutIndex: u32, cascadeIndex: u32) {
	if (layoutIndex >= arrayLength(&layouts)) {
		return;
	}
	let pageLayout = layouts[layoutIndex];
	if (pageLayout.feedbackMode == 0u || cascadeIndex >= pageLayout.cascadeCount) {
		return;
	}
	let matrixIndex = layoutIndex * 4u + cascadeIndex;
	if (matrixIndex >= arrayLength(&cascadeViewProjections)) {
		return;
	}
	let clip = cascadeViewProjections[matrixIndex] * vec4<f32>(worldPosition, 1.0);
	if (abs(clip.w) <= EPSILON) {
		return;
	}
	let ndc = clip.xyz / vec3<f32>(clip.w);
	if (
		ndc.x < -1.0 || ndc.x > 1.0 ||
		ndc.y < -1.0 || ndc.y > 1.0 ||
		ndc.z < -1.0 || ndc.z > 1.0
	) {
		return;
	}
	let gridSize = max(pageLayout.pageGridSize, 1u);
	let u = clamp(ndc.x * 0.5 + 0.5, 0.0, 0.999999);
	let v = clamp(0.5 - ndc.y * 0.5, 0.0, 0.999999);
	let pageX = min(u32(floor(u * f32(gridSize))), gridSize - 1u);
	let pageY = min(u32(floor(v * f32(gridSize))), gridSize - 1u);
	let tableIndex =
		pageLayout.pageTableBase +
		cascadeIndex * pageLayout.pageTableCascadeStride +
		pageY * gridSize +
		pageX;
	if (
		tableIndex < params.pageTableLength &&
		tableIndex < params.feedbackFlagCount &&
		tableIndex < arrayLength(&pageTable) &&
		tableIndex < arrayLength(&nextFeedbackFlags)
	) {
		nextFeedbackFlags[tableIndex] = params.frameId;
	}
}

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let STRIDE: u32 = 8u;
	let pixelCoord = globalId.xy * STRIDE;
	if (pixelCoord.x >= params.width || pixelCoord.y >= params.height) {
		return;
	}
	if (
		params.pageTableLength == 0u ||
		params.layoutCount == 0u ||
		arrayLength(&nextFeedbackFlags) == 0u ||
		arrayLength(&inverseViewProjection) == 0u
	) {
		return;
	}
	let world = reconstructWorldPosition(pixelCoord);
	if (world.w <= 0.5) {
		return;
	}
	let layoutCount = min(params.layoutCount, arrayLength(&layouts));
	for (var layoutIndex = 0u; layoutIndex < layoutCount; layoutIndex = layoutIndex + 1u) {
		let cascadeCount = min(layouts[layoutIndex].cascadeCount, 4u);
		for (
			var cascadeIndex = 0u;
			cascadeIndex < cascadeCount;
			cascadeIndex = cascadeIndex + 1u
		) {
			markFeedbackPage(world.xyz, layoutIndex, cascadeIndex);
		}
	}
}
