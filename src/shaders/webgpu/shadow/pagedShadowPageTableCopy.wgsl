struct PagedShadowCopyParams {
	pageTableLength: u32,
	pageGridSize: u32,
	_pad0: u32,
	_pad1: u32,
}

@group(0) @binding(0) var<uniform> params: PagedShadowCopyParams;
@group(0) @binding(1) var<storage, read> pageTableBuffer: array<u32>;
@group(0) @binding(2) var pageTableTexture: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(64)
fn csMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
	let index = globalId.x;
	if (index >= params.pageTableLength) {
		return;
	}
	let value = pageTableBuffer[index];
	let gridSize = max(params.pageGridSize, 1u);
	let x = index % gridSize;
	let y = index / gridSize;
	textureStore(pageTableTexture, vec2<i32>(i32(x), i32(y)), vec4<u32>(value, 0u, 0u, 0u));
}
