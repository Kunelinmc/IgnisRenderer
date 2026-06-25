struct PagedShadowFeedbackParams {
	pageTableLength: u32,
	width: u32,
	height: u32,
	_pad0: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowFeedbackParams;
@group(0) @binding(1) var<storage, read_write> nextFeedbackFlags: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	if (globalId.x >= params.width || globalId.y >= params.height) {
		return;
	}
	if (params.pageTableLength == 0u || arrayLength(&nextFeedbackFlags) == 0u) {
		return;
	}

	// V2 reserves this pass for screen-space feedback. The runtime currently
	// consumes previous feedback and conservative requests; this keeps the pass
	// graph-resident until full G-buffer reconstruction is wired in.
	if (globalId.x == 0u && globalId.y == 0u) {
		nextFeedbackFlags[0] = nextFeedbackFlags[0];
	}
}
