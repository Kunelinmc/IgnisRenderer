struct DirectionalLightData {
	direction: vec4<f32>,
	color: vec4<f32>,
}

struct PointLightData {
	positionRange: vec4<f32>,
	color: vec4<f32>,
}

struct SpotLightData {
	positionRange: vec4<f32>,
	directionOuter: vec4<f32>,
	colorInner: vec4<f32>,
}

struct ShadowData {
	viewProjection: mat4x4<f32>,
	paramsA: vec4<f32>,
	paramsB: vec4<f32>,
}

struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	skyboxBasisRight: vec4<f32>,
	skyboxBasisUp: vec4<f32>,
	skyboxBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
	directionalLights: array<DirectionalLightData, 4>,
	pointLights: array<PointLightData, 4>,
	spotLights: array<SpotLightData, 4>,
	directionalShadows: array<ShadowData, 4>,
	spotShadows: array<ShadowData, 4>,
	shAmbientCoeffs: array<vec4<f32>, 16>,
}

struct TraceParams {
	invHalfSize: vec2<f32>,
	maxDistance: f32,
	thickness: f32,
	stride: f32,
	intensity: f32,
	maxRoughness: f32,
	edgeFade: f32,
	maxSteps: f32,
	binarySearchSteps: f32,
	maxMip: f32,
	historyWeight: f32,
	historyValid: f32,
	depthThreshold: f32,
}

@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var gNormalRoughMetal: texture_2d<f32>;
@group(0) @binding(2) var gMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var hiZ: texture_2d<f32>;
@group(0) @binding(4) var ssrHistory: texture_2d<f32>;
@group(0) @binding(5) var motionHistory: texture_2d<f32>;
@group(0) @binding(6) var linearSampler: sampler;
@group(0) @binding(7) var<uniform> traceParams: TraceParams;
@group(0) @binding(8) var outSSR: texture_storage_2d<rgba16float, write>;

@group(1) @binding(0) var<uniform> frame: FrameUniforms;

struct ComposeParams {
	invFullSize: vec2<f32>,
	_pad0: vec2<f32>,
}

@group(0) @binding(0) var composeScene: texture_2d<f32>;
@group(0) @binding(1) var composeSSR: texture_2d<f32>;
@group(0) @binding(2) var composeMotionDepth: texture_2d<f32>;
@group(0) @binding(3) var composeSampler: sampler;
@group(0) @binding(4) var<uniform> composeParams: ComposeParams;
@group(0) @binding(5) var composeOut: texture_storage_2d<rgba16float, write>;

fn decodeNormal(encoded: vec2<f32>) -> vec3<f32> {
	let xy = encoded * 2.0 - vec2<f32>(1.0, 1.0);
	let z2 = max(1.0 - dot(xy, xy), 0.0);
    let vn = vec3<f32>(xy, sqrt(z2));
    
    // Transform back to World Space for tracing consistency
    let right = frame.skyboxBasisRight.xyz;
    let up = frame.skyboxBasisUp.xyz;
    let backward = frame.skyboxBasisBackward.xyz;
    return normalize(right * vn.x + up * vn.y + backward * vn.z);
}

fn getPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let right = frame.skyboxBasisRight.xyz;
    let up = frame.skyboxBasisUp.xyz;
    let backward = frame.skyboxBasisBackward.xyz;
    let tanHalfFov = frame.skyboxBasisRight.w;
    let aspect = frame.skyboxBasisUp.w;
    
    let cx = ndc.x * aspect * tanHalfFov * depth;
    let cy = ndc.y * tanHalfFov * depth;
    return frame.cameraPosition.xyz + right * cx + up * cy - backward * depth;
}

fn worldToUv(worldPos: vec3<f32>) -> vec2<f32> {
    let rel = worldPos - frame.cameraPosition.xyz;
    let right = frame.skyboxBasisRight.xyz;
    let up = frame.skyboxBasisUp.xyz;
    let backward = frame.skyboxBasisBackward.xyz;
    let tanHalfFov = frame.skyboxBasisRight.w;
    let aspect = frame.skyboxBasisUp.w;
    
    let depth = dot(rel, -backward);
    if (depth <= 1e-4) { return vec2<f32>(-1.0); }
    
    let cx = dot(rel, right) / (depth * aspect * tanHalfFov);
    let cy = dot(rel, up) / (depth * tanHalfFov);
    
    return vec2<f32>(cx * 0.5 + 0.5, 0.5 - cy * 0.5);
}

@compute @workgroup_size(8, 8, 1)
fn csTrace(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(outSSR);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * traceParams.invHalfSize;
	let g = textureSampleLevel(gNormalRoughMetal, linearSampler, uv, 0.0);
	let depth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).z;
	let roughness = clamp(g.z, 0.0, 1.0);
	let metalness = clamp(g.w, 0.0, 1.0);
	if (depth <= 0.0 || roughness > traceParams.maxRoughness) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }
	
    let worldPos = getPosition(uv, depth);
    let worldNormal = decodeNormal(g.xy);
    let worldViewDir = normalize(worldPos - frame.cameraPosition.xyz);
    let reflectionDir = reflect(worldViewDir, worldNormal);
    
    // Safety check: skip if reflection points back into surface
    if (dot(reflectionDir, worldNormal) < 0.0) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }

	var hitUv = uv;
	var hit = false;
	var t = max(traceParams.stride, 0.25);
	var hitT = t;
	var missT = 0.0;
	let maxSteps = i32(clamp(traceParams.maxSteps, 1.0, 128.0));
	for (var i: i32 = 0; i < 128; i = i + 1) {
		if (i >= maxSteps) { break; }
		let samplePos = worldPos + reflectionDir * t;
		let suv = worldToUv(samplePos);
		if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { break; }
		
        let rayDepth = dot(samplePos - frame.cameraPosition.xyz, -frame.skyboxBasisBackward.xyz);
		let mip = clamp(i32(log2(max(t * 0.1, 1.0))), 0, i32(traceParams.maxMip));
		let sceneDepth = textureSampleLevel(hiZ, linearSampler, suv, f32(mip)).x;
		
        if (sceneDepth > 0.0 && rayDepth >= sceneDepth - traceParams.thickness && rayDepth < sceneDepth + 2.0) {
			hit = true;
			hitUv = suv;
			hitT = t;
			break;
		}
		missT = t;
		t += traceParams.stride;
		if (t > traceParams.maxDistance) { break; }
	}

	if (!hit) { textureStore(outSSR, vec2<i32>(gid.xy), vec4<f32>(0.0)); return; }
	
    let refineCount = i32(clamp(traceParams.binarySearchSteps, 0.0, 16.0));
	for (var j: i32 = 0; j < 16; j = j + 1) {
		if (j >= refineCount) { break; }
		let midT = (missT + hitT) * 0.5;
		let refinePos = worldPos + reflectionDir * midT;
		let refineUv = worldToUv(refinePos);
		if (refineUv.x < 0.0 || refineUv.x > 1.0 || refineUv.y < 0.0 || refineUv.y > 1.0) {
			hitT = midT;
			continue;
		}
		let rayDepth = dot(refinePos - frame.cameraPosition.xyz, -frame.skyboxBasisBackward.xyz);
		let sceneDepth = textureSampleLevel(hiZ, linearSampler, refineUv, 0.0).x;
		if (sceneDepth > 0.0 && rayDepth >= sceneDepth - traceParams.thickness) {
			hitT = midT;
			hitUv = refineUv;
		} else {
			missT = midT;
		}
	}

	let hitColor = textureSampleLevel(sceneColor, linearSampler, hitUv, 0.0).rgb;
	let edgeDistance = min(min(hitUv.x, 1.0 - hitUv.x), min(hitUv.y, 1.0 - hitUv.y));
	let edge = clamp(edgeDistance / max(traceParams.edgeFade, 1e-4), 0.0, 1.0);
	
	let reflectivity = mix(0.1, 1.0, metalness);
	let weight = clamp(traceParams.intensity * edge * reflectivity * (1.0 - roughness), 0.0, 1.0);
	
    let motion = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).xy;
	let prevUv = uv - vec2<f32>(motion.x * 0.5, -motion.y * 0.5);
	let hist = textureSampleLevel(ssrHistory, linearSampler, prevUv, 0.0);
	let currDepth = textureSampleLevel(gMotionDepth, linearSampler, uv, 0.0).z;
	let prevDepth = textureSampleLevel(motionHistory, linearSampler, prevUv, 0.0).z;
	let relDepth = abs(currDepth - prevDepth) / max(max(currDepth, prevDepth), 1e-4);
	let historyOk = traceParams.historyValid > 0.5 && prevUv.x >= 0.0 && prevUv.x <= 1.0 && prevUv.y >= 0.0 && prevUv.y <= 1.0 && relDepth <= traceParams.depthThreshold;
	let blend = select(0.0, clamp(traceParams.historyWeight, 0.0, 0.95), historyOk);
	
    textureStore(outSSR, vec2<i32>(gid.xy), mix(vec4<f32>(max(hitColor * weight, vec3<f32>(0.0)), weight), hist, blend));
}

@compute @workgroup_size(8, 8, 1)
fn csCompose(@builtin(global_invocation_id) gid: vec3<u32>) {
	let size = textureDimensions(composeOut);
	if (gid.x >= size.x || gid.y >= size.y) { return; }
	let coord = vec2<i32>(gid.xy);
	let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) * composeParams.invFullSize;
	let scene = textureLoad(composeScene, coord, 0);
	let centerDepth = textureSampleLevel(composeMotionDepth, composeSampler, uv, 0.0).z;
	let step = composeParams.invFullSize;
	let taps = array<vec2<f32>, 5>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(step.x, 0.0),
		vec2<f32>(-step.x, 0.0),
		vec2<f32>(0.0, step.y),
		vec2<f32>(0.0, -step.y)
	);
	var ssrSum = vec4<f32>(0.0);
	var weightSum = 0.0;
	for (var i: i32 = 0; i < 5; i = i + 1) {
		let sampleUv = uv + taps[i];
		let sampleDepth = textureSampleLevel(composeMotionDepth, composeSampler, sampleUv, 0.0).z;
		let depthWeight = exp(-abs(sampleDepth - centerDepth) * 48.0);
		let ssrTap = textureSampleLevel(composeSSR, composeSampler, sampleUv, 0.0);
		ssrSum += ssrTap * depthWeight;
		weightSum += depthWeight;
	}
	let ssr = select(textureSampleLevel(composeSSR, composeSampler, uv, 0.0), ssrSum / max(weightSum, 1e-4), weightSum > 0.0);
	textureStore(
		composeOut,
		coord,
		vec4<f32>(max(scene.rgb + ssr.rgb, vec3<f32>(0.0)), scene.a)
	);
}
