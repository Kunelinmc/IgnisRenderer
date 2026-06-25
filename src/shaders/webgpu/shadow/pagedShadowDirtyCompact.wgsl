const PAGED_SHADOW_RESIDENCY_UINTS: u32 = 8u;
const PAGED_SHADOW_DIRTY_RECORD_UINTS: u32 = 8u;
const PAGED_SHADOW_NON_RESIDENT: u32 = 0xffffffffu;

struct PagedShadowDirtyParams {
	physicalPageCount: u32,
	maxDirtyPages: u32,
	physicalGridSize: u32,
	pageSize: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowDirtyParams;
@group(0) @binding(1) var<storage, read_write> residencyState: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> dirtyPhysicalPages: array<u32>;

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let physicalPageIndex = globalId.x;
	if (physicalPageIndex >= params.physicalPageCount) {
		return;
	}
	let base = physicalPageIndex * PAGED_SHADOW_RESIDENCY_UINTS;
	if (base + 7u >= arrayLength(&residencyState)) {
		return;
	}
	if (residencyState[base] == PAGED_SHADOW_NON_RESIDENT || residencyState[base + 3u] == 0u) {
		return;
	}
	let dirtyIndex = atomicAdd(&counters[1], 1u);
	if (dirtyIndex >= params.maxDirtyPages) {
		return;
	}
	let outBase = dirtyIndex * PAGED_SHADOW_DIRTY_RECORD_UINTS;
	if (outBase + 7u >= arrayLength(&dirtyPhysicalPages)) {
		return;
	}
	dirtyPhysicalPages[outBase] = physicalPageIndex;
	dirtyPhysicalPages[outBase + 1u] = residencyState[base];
	dirtyPhysicalPages[outBase + 2u] = residencyState[base + 4u];
	dirtyPhysicalPages[outBase + 3u] = residencyState[base + 5u];
	dirtyPhysicalPages[outBase + 4u] = residencyState[base + 6u];
	dirtyPhysicalPages[outBase + 5u] = (physicalPageIndex % params.physicalGridSize) * params.pageSize;
	dirtyPhysicalPages[outBase + 6u] = (physicalPageIndex / params.physicalGridSize) * params.pageSize;
	dirtyPhysicalPages[outBase + 7u] = max(residencyState[base + 7u], 1u);
	residencyState[base + 3u] = 0u;
}
