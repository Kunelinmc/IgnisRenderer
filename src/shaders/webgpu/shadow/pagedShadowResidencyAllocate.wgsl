const PAGED_SHADOW_NON_RESIDENT: u32 = 0xffffffffu;
const PAGED_SHADOW_REQUEST_RECORD_UINTS: u32 = 8u;
const PAGED_SHADOW_RESIDENCY_UINTS: u32 = 8u;

struct PagedShadowAllocationParams {
	frameId: u32,
	requestCountLimit: u32,
	physicalPageCount: u32,
	maxPagesPerFrame: u32,
	cacheFrames: u32,
	pageTableLength: u32,
	_pad0: u32,
	_pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowAllocationParams;
@group(0) @binding(1) var<storage, read_write> pageTable: array<u32>;
@group(0) @binding(2) var<storage, read_write> residencyState: array<u32>;
@group(0) @binding(3) var<storage, read> compactedRequests: array<u32>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;

fn residencyBase(physicalPageIndex: u32) -> u32 {
	return physicalPageIndex * PAGED_SHADOW_RESIDENCY_UINTS;
}

@compute @workgroup_size(1)
fn csMain() {
	let requestCount = min(atomicLoad(&counters[0]), params.requestCountLimit);
	var allocatedThisFrame = 0u;
	for (var requestIndex = 0u; requestIndex < requestCount; requestIndex = requestIndex + 1u) {
		let requestBase = requestIndex * PAGED_SHADOW_REQUEST_RECORD_UINTS;
		if (requestBase >= arrayLength(&compactedRequests)) {
			break;
		}
		let tableIndex = compactedRequests[requestBase];
		if (tableIndex >= params.pageTableLength || tableIndex >= arrayLength(&pageTable)) {
			continue;
		}

		var physicalPage = pageTable[tableIndex];
		if (physicalPage != PAGED_SHADOW_NON_RESIDENT && physicalPage < params.physicalPageCount) {
			let base = residencyBase(physicalPage);
			if (base + 7u < arrayLength(&residencyState)) {
				residencyState[base] = tableIndex;
				residencyState[base + 1u] = params.frameId;
				residencyState[base + 2u] = 1u;
			}
			continue;
		}
		if (allocatedThisFrame >= params.maxPagesPerFrame) {
			continue;
		}

		var selected = PAGED_SHADOW_NON_RESIDENT;
		for (var pageIndex = 0u; pageIndex < params.physicalPageCount; pageIndex = pageIndex + 1u) {
			let base = residencyBase(pageIndex);
			if (base + 7u >= arrayLength(&residencyState)) {
				continue;
			}
			if (residencyState[base] == PAGED_SHADOW_NON_RESIDENT) {
				selected = pageIndex;
				break;
			}
		}
		if (selected == PAGED_SHADOW_NON_RESIDENT) {
			var oldestFrame = 0xffffffffu;
			for (var pageIndex = 0u; pageIndex < params.physicalPageCount; pageIndex = pageIndex + 1u) {
				let base = residencyBase(pageIndex);
				if (base + 7u >= arrayLength(&residencyState)) {
					continue;
				}
				let residentTableIndex = residencyState[base];
				let lastUsedFrame = residencyState[base + 1u];
				if (residentTableIndex == PAGED_SHADOW_NON_RESIDENT) {
					selected = pageIndex;
					break;
				}
				if (lastUsedFrame < oldestFrame) {
					oldestFrame = lastUsedFrame;
					selected = pageIndex;
				}
			}
		}
		if (selected == PAGED_SHADOW_NON_RESIDENT) {
			continue;
		}

		let selectedBase = residencyBase(selected);
		let oldTableIndex = residencyState[selectedBase];
		if (oldTableIndex != PAGED_SHADOW_NON_RESIDENT && oldTableIndex < arrayLength(&pageTable)) {
			pageTable[oldTableIndex] = PAGED_SHADOW_NON_RESIDENT;
		}
		residencyState[selectedBase] = tableIndex;
		residencyState[selectedBase + 1u] = params.frameId;
		residencyState[selectedBase + 2u] = 1u;
		residencyState[selectedBase + 3u] = 1u;
		residencyState[selectedBase + 4u] = compactedRequests[requestBase + 1u];
		residencyState[selectedBase + 5u] = compactedRequests[requestBase + 2u];
		residencyState[selectedBase + 6u] = compactedRequests[requestBase + 3u];
		residencyState[selectedBase + 7u] = compactedRequests[requestBase + 4u];
		pageTable[tableIndex] = selected;
		allocatedThisFrame = allocatedThisFrame + 1u;
	}

	for (var pageIndex = 0u; pageIndex < params.physicalPageCount; pageIndex = pageIndex + 1u) {
		let base = residencyBase(pageIndex);
		if (base + 7u >= arrayLength(&residencyState)) {
			continue;
		}
		let tableIndex = residencyState[base];
		if (tableIndex == PAGED_SHADOW_NON_RESIDENT) {
			continue;
		}
		let lastUsedFrame = residencyState[base + 1u];
		if (params.frameId > lastUsedFrame + params.cacheFrames) {
			if (tableIndex < arrayLength(&pageTable)) {
				pageTable[tableIndex] = PAGED_SHADOW_NON_RESIDENT;
			}
			residencyState[base] = PAGED_SHADOW_NON_RESIDENT;
			residencyState[base + 2u] = 0u;
			residencyState[base + 3u] = 0u;
		}
	}
}
