@group(2) @binding(0) var<storage, read_write> probeOut: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probeCrossing(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x >= 1000u) { return; }
	let jitter = (f32(gid.x) + 0.5) / 1000.0;
	let hit = traceHiZ(
		vec3<f32>(0.0, 0.0, -2.0), vec3<f32>(0.0, 0.0, -1.0), 1.0, jitter
	);
	probeOut[gid.x] = vec4<f32>(select(0.0, 1.0, hit.hit), hit.t, hit.hitUv);
}

@compute @workgroup_size(64)
fn probeDepthEdge(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x >= 1000u) { return; }
	let jitter = (f32(gid.x) + 0.5) / 1000.0;
	let hit = traceHiZ(
		vec3<f32>(-0.5, 0.0, -2.0), normalize(vec3<f32>(0.2, 0.0, -1.0)), 1.0, jitter
	);
	probeOut[gid.x] = vec4<f32>(select(0.0, 1.0, hit.hit), hit.t, hit.hitUv);
}
