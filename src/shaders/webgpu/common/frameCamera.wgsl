// Camera matrices use unjittered OpenGL clip coordinates; raster UVs start at top-left.
struct FrameCameraUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	inverseView: mat4x4<f32>,
	inverseProjection: mat4x4<f32>,
	jitterNdc: vec2<f32>,
	prevJitterNdc: vec2<f32>,
}

fn cameraPosition(inverseView: mat4x4<f32>) -> vec3<f32> {
	return inverseView[3].xyz;
}

fn cameraRight(inverseView: mat4x4<f32>) -> vec3<f32> {
	return inverseView[0].xyz;
}

fn cameraUp(inverseView: mat4x4<f32>) -> vec3<f32> {
	return inverseView[1].xyz;
}

fn cameraBackward(inverseView: mat4x4<f32>) -> vec3<f32> {
	return inverseView[2].xyz;
}

fn cameraWorldToViewDirection(
	inverseView: mat4x4<f32>, direction: vec3<f32>
) -> vec3<f32> {
	return vec3<f32>(
		dot(direction, cameraRight(inverseView)),
		dot(direction, cameraUp(inverseView)),
		dot(direction, cameraBackward(inverseView))
	);
}

fn cameraViewToWorldDirection(
	inverseView: mat4x4<f32>, direction: vec3<f32>
) -> vec3<f32> {
	return (inverseView * vec4<f32>(direction, 0.0)).xyz;
}

fn cameraWorldToView(inverseView: mat4x4<f32>, position: vec3<f32>) -> vec3<f32> {
	return cameraWorldToViewDirection(inverseView, position - cameraPosition(inverseView));
}

fn cameraLinearDepth(inverseView: mat4x4<f32>, position: vec3<f32>) -> f32 {
	return -cameraWorldToView(inverseView, position).z;
}

fn cameraRasterUvToNdc(jitterNdc: vec2<f32>, uv: vec2<f32>) -> vec2<f32> {
	return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) - jitterNdc;
}

fn cameraUnprojectView(
	inverseProjection: mat4x4<f32>, ndc: vec2<f32>, clipZ: f32
) -> vec3<f32> {
	let homogeneous = inverseProjection * vec4<f32>(ndc, clipZ, 1.0);
	// Preserve signed homogeneous division, including points behind the camera.
	return homogeneous.xyz / homogeneous.w;
}

fn cameraReconstructHardwareDepth(
	inverseProjection: mat4x4<f32>, jitterNdc: vec2<f32>,
	uv: vec2<f32>, hardwareDepth: f32
) -> vec3<f32> {
	return cameraUnprojectView(
		inverseProjection, cameraRasterUvToNdc(jitterNdc, uv), hardwareDepth * 2.0 - 1.0
	);
}

fn cameraReconstructHardwareWorldPosition(
	inverseView: mat4x4<f32>, inverseProjection: mat4x4<f32>, jitterNdc: vec2<f32>,
	uv: vec2<f32>, hardwareDepth: f32
) -> vec3<f32> {
	let viewPosition = cameraReconstructHardwareDepth(inverseProjection, jitterNdc, uv, hardwareDepth);
	return (inverseView * vec4<f32>(viewPosition, 1.0)).xyz;
}

fn cameraReconstructViewPosition(
	inverseProjection: mat4x4<f32>, jitterNdc: vec2<f32>,
	uv: vec2<f32>, linearDepth: f32
) -> vec3<f32> {
	let ndc = cameraRasterUvToNdc(jitterNdc, uv);
	let start = inverseProjection * vec4<f32>(ndc, -1.0, 1.0);
	let end = inverseProjection * vec4<f32>(ndc, 0.0, 1.0);
	// Intersect viewZ = -depth before dividing: an oblique clip plane may have w = 0.
	let ray = end.xyz * start.w - start.xyz * end.w;
	let originNumerator = start.xyz * end.z - end.xyz * start.z;
	// Keep the denominator depth-independent to avoid cancelling large f32 depth terms.
	return (originNumerator - ray * linearDepth) / ray.z;
}

fn cameraReconstructWorldPosition(
	inverseView: mat4x4<f32>, inverseProjection: mat4x4<f32>, jitterNdc: vec2<f32>,
	uv: vec2<f32>, linearDepth: f32
) -> vec3<f32> {
	let viewPosition = cameraReconstructViewPosition(inverseProjection, jitterNdc, uv, linearDepth);
	return (inverseView * vec4<f32>(viewPosition, 1.0)).xyz;
}

fn cameraWorldToRasterUv(
	viewProjection: mat4x4<f32>, jitterNdc: vec2<f32>, position: vec3<f32>
) -> vec2<f32> {
	let clip = viewProjection * vec4<f32>(position, 1.0);
	let ndc = clip.xy / clip.w + jitterNdc;
	return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

fn cameraWorldRayOrigin(
	inverseView: mat4x4<f32>, inverseProjection: mat4x4<f32>,
	jitterNdc: vec2<f32>, uv: vec2<f32>
) -> vec3<f32> {
	return cameraReconstructWorldPosition(inverseView, inverseProjection, jitterNdc, uv, 0.0);
}

fn cameraWorldRayDirection(
	inverseView: mat4x4<f32>, inverseProjection: mat4x4<f32>,
	jitterNdc: vec2<f32>, uv: vec2<f32>
) -> vec3<f32> {
	let ndc = cameraRasterUvToNdc(jitterNdc, uv);
	let start = inverseProjection * vec4<f32>(ndc, -1.0, 1.0);
	let end = inverseProjection * vec4<f32>(ndc, 0.0, 1.0);
	let ray = end.xyz * start.w - start.xyz * end.w;
	// Clip planes can straddle infinity; orient the homogeneous ray toward view -Z.
	return normalize(cameraViewToWorldDirection(inverseView, ray / -ray.z));
}
