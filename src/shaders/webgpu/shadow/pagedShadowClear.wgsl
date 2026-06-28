struct PagedShadowClearParams {
	physicalPageCount: u32,
	pageSize: u32,
	physicalGridSize: u32,
	_pad0: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowClearParams;
@group(0) @binding(1) var<storage, read> dirtyPhysicalPages: array<u32>;

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
}

@vertex
fn vsMain(
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
	var output: VertexOutput;

	// Quad vertices (2 triangles)
	var positions = array<vec2<f32>, 6>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>( 1.0, -1.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>(-1.0,  1.0),
		vec2<f32>( 1.0, -1.0),
		vec2<f32>( 1.0,  1.0)
	);

	let quadPos = positions[vertexIndex];

	let dirtyBase = instanceIndex * 8u; // DIRTY_PHYSICAL_PAGE_RECORD_UINTS = 8u
	if (dirtyBase + 7u >= arrayLength(&dirtyPhysicalPages)) {
		output.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
		return output;
	}

	let viewportX = f32(dirtyPhysicalPages[dirtyBase + 5u]);
	let viewportY = f32(dirtyPhysicalPages[dirtyBase + 6u]);
	let pageSize = f32(params.pageSize);
	let atlasSize = f32(params.physicalGridSize * params.pageSize);

	let u = (viewportX + (quadPos.x * 0.5 + 0.5) * pageSize) / atlasSize;
	let v = (viewportY + (0.5 - quadPos.y * 0.5) * pageSize) / atlasSize;
	let ndcX = u * 2.0 - 1.0;
	let ndcY = 1.0 - v * 2.0;

	output.position = vec4<f32>(ndcX, ndcY, 1.0, 1.0); // Z = 1.0 clears depth to 1.0
	return output;
}
