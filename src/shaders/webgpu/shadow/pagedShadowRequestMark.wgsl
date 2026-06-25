const PAGED_SHADOW_REQUEST_SOURCE_CONSERVATIVE: u32 = 1u;
const PAGED_SHADOW_REQUEST_SOURCE_FEEDBACK: u32 = 2u;

struct PagedShadowRequestParams {
	pageTableLength: u32,
	casterCount: u32,
	layoutCount: u32,
	frameId: u32,
	conservativeWarmup: u32,
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

struct PagedShadowCasterBounds {
	centerRadius: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: PagedShadowRequestParams;
@group(0) @binding(1) var<storage, read_write> pageRequestFlags: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> feedbackFlags: array<u32>;
@group(0) @binding(3) var<storage, read> layouts: array<PagedShadowLayoutData>;
@group(0) @binding(4) var<storage, read> casterBounds: array<PagedShadowCasterBounds>;
@group(0) @binding(5) var<storage, read> cascadeViewProjections: array<mat4x4<f32>>;

fn projectPoint(matrix: mat4x4<f32>, point: vec3<f32>) -> vec3<f32> {
	let clip = matrix * vec4<f32>(point, 1.0);
	if (abs(clip.w) <= 0.000001) {
		return vec3<f32>(2.0, 2.0, 2.0);
	}
	return clip.xyz / vec3<f32>(clip.w);
}

fn markFeedbackRequests(index: u32) {
	if (index >= min(params.feedbackFlagCount, params.pageTableLength)) {
		return;
	}
	if (feedbackFlags[index] != 0u && index < arrayLength(&pageRequestFlags)) {
		atomicOr(&pageRequestFlags[index], PAGED_SHADOW_REQUEST_SOURCE_FEEDBACK);
	}
}

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let index = globalId.x;
	if (index < params.pageTableLength) {
		markFeedbackRequests(index);
	}
	if (params.conservativeWarmup == 0u || index >= params.casterCount) {
		return;
	}

	let bounds = casterBounds[index].centerRadius;
	let center = bounds.xyz;
	let radius = max(bounds.w, 0.0);
	let corners = array<vec3<f32>, 8>(
		center + vec3<f32>(-radius, -radius, -radius),
		center + vec3<f32>( radius, -radius, -radius),
		center + vec3<f32>(-radius,  radius, -radius),
		center + vec3<f32>( radius,  radius, -radius),
		center + vec3<f32>(-radius, -radius,  radius),
		center + vec3<f32>( radius, -radius,  radius),
		center + vec3<f32>(-radius,  radius,  radius),
		center + vec3<f32>( radius,  radius,  radius)
	);

	for (var layoutIndex = 0u; layoutIndex < params.layoutCount; layoutIndex = layoutIndex + 1u) {
		let pagedLayout = layouts[layoutIndex];
		let gridSize = max(pagedLayout.pageGridSize, 1u);
		for (var cascadeIndex = 0u; cascadeIndex < pagedLayout.cascadeCount; cascadeIndex = cascadeIndex + 1u) {
			let matrixIndex = layoutIndex * 4u + cascadeIndex;
			if (matrixIndex >= arrayLength(&cascadeViewProjections)) {
				continue;
			}
			let viewProjection = cascadeViewProjections[matrixIndex];
			var minUv = vec2<f32>(1.0, 1.0);
			var maxUv = vec2<f32>(0.0, 0.0);
			var hasProjected = false;
			for (var cornerIndex = 0u; cornerIndex < 8u; cornerIndex = cornerIndex + 1u) {
				let ndc = projectPoint(viewProjection, corners[cornerIndex]);
				if (ndc.z < -1.0 || ndc.z > 1.0) {
					continue;
				}
				let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
				minUv = min(minUv, uv);
				maxUv = max(maxUv, uv);
				hasProjected = true;
			}
			if (!hasProjected || maxUv.x < 0.0 || minUv.x > 1.0 || maxUv.y < 0.0 || minUv.y > 1.0) {
				continue;
			}

			let clampedMin = clamp(minUv, vec2<f32>(0.0), vec2<f32>(1.0));
			let clampedMax = clamp(maxUv, vec2<f32>(0.0), vec2<f32>(1.0));
			let minPage = vec2<u32>(
				u32(clamp(floor(clampedMin.x * f32(gridSize)), 0.0, f32(gridSize - 1u))),
				u32(clamp(floor(clampedMin.y * f32(gridSize)), 0.0, f32(gridSize - 1u)))
			);
			let maxPage = vec2<u32>(
				u32(clamp(floor(clampedMax.x * f32(gridSize)), 0.0, f32(gridSize - 1u))),
				u32(clamp(floor(clampedMax.y * f32(gridSize)), 0.0, f32(gridSize - 1u)))
			);
			for (var pageY = minPage.y; pageY <= maxPage.y; pageY = pageY + 1u) {
				for (var pageX = minPage.x; pageX <= maxPage.x; pageX = pageX + 1u) {
					let tableIndex =
						pagedLayout.pageTableBase +
						cascadeIndex * pagedLayout.pageTableCascadeStride +
						pageY * gridSize +
						pageX;
					if (tableIndex < params.pageTableLength) {
						atomicOr(&pageRequestFlags[tableIndex], PAGED_SHADOW_REQUEST_SOURCE_CONSERVATIVE);
					}
				}
			}
		}
	}
}
