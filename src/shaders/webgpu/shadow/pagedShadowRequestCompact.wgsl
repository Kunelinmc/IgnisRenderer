const PAGED_SHADOW_REQUEST_RECORD_UINTS: u32 = 8u;

struct PagedShadowCompactParams {
	pageTableLength: u32,
	maxRequests: u32,
	layoutCount: u32,
	_pad0: u32,
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

struct PageTableAddress {
	matrixIndex: u32,
	pageX: u32,
	pageY: u32,
	pageGridSize: u32,
	priority: u32,
	valid: bool,
}

@group(0) @binding(0) var<uniform> params: PagedShadowCompactParams;
@group(0) @binding(1) var<storage, read> pageRequestFlags: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> compactedRequests: array<u32>;
@group(0) @binding(4) var<storage, read> layouts: array<PagedShadowLayoutData>;

fn resolvePageTableAddress(tableIndex: u32) -> PageTableAddress {
	let layoutCount = min(params.layoutCount, arrayLength(&layouts));
	for (var layoutIndex = 0u; layoutIndex < layoutCount; layoutIndex = layoutIndex + 1u) {
		let pageLayout = layouts[layoutIndex];
		let gridSize = max(pageLayout.pageGridSize, 1u);
		let cascadeStride = max(pageLayout.pageTableCascadeStride, gridSize * gridSize);
		let cascadeCount = max(pageLayout.cascadeCount, 1u);
		let layoutBase = pageLayout.pageTableBase;
		let layoutPageCount = cascadeStride * cascadeCount;
		if (tableIndex < layoutBase || tableIndex >= layoutBase + layoutPageCount) {
			continue;
		}
		let localIndex = tableIndex - layoutBase;
		let cascadeIndex = localIndex / cascadeStride;
		if (cascadeIndex >= cascadeCount) {
			break;
		}
		let pageIndex = localIndex - cascadeIndex * cascadeStride;
		return PageTableAddress(
			layoutIndex * 4u + cascadeIndex,
			pageIndex % gridSize,
			(pageIndex / gridSize) % gridSize,
			gridSize,
			pageLayout.priorityBase + (cascadeCount - min(cascadeIndex, cascadeCount - 1u)),
			true
		);
	}
	return PageTableAddress(0u, 0u, 0u, 1u, 0u, false);
}

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let tableIndex = globalId.x;
	if (tableIndex >= params.pageTableLength) {
		return;
	}
	let flags = pageRequestFlags[tableIndex];
	if (flags == 0u) {
		return;
	}
	let address = resolvePageTableAddress(tableIndex);
	if (!address.valid) {
		return;
	}
	let requestIndex = atomicAdd(&counters[0], 1u);
	if (requestIndex >= params.maxRequests) {
		return;
	}
	let base = requestIndex * PAGED_SHADOW_REQUEST_RECORD_UINTS;
	if (base + 7u >= arrayLength(&compactedRequests)) {
		return;
	}
	compactedRequests[base] = tableIndex;
	compactedRequests[base + 1u] = address.matrixIndex;
	compactedRequests[base + 2u] = address.pageX;
	compactedRequests[base + 3u] = address.pageY;
	compactedRequests[base + 4u] = flags;
	compactedRequests[base + 5u] = address.priority;
	compactedRequests[base + 6u] = address.pageGridSize;
	compactedRequests[base + 7u] = 0u;
}
