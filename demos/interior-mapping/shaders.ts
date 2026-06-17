export const INTERIOR_MAPPING_VERTEX_WGSL = `
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
}

struct ModelUniforms {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) uv0: vec2<f32>,
	@location(2) normal: vec3<f32>,
	@location(3) tangent: vec4<f32>,
}

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv0: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
}

@vertex
fn vsMain(input: VertexInput) -> VertexOutput {
	var output: VertexOutput;
	let worldPos = model.modelMatrix * vec4<f32>(input.position, 1.0);
	output.position = frame.viewProjection * worldPos;
	output.worldPosition = worldPos.xyz;
	output.worldNormal = normalize((model.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);

	let tangentWorld = (model.normalMatrix * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
	output.worldTangent = vec4<f32>(
		normalize(tangentWorld),
		input.tangent.w
	);

	output.uv0 = input.uv0;
	return output;
}
`;

export const INTERIOR_MAPPING_FRAGMENT_WGSL = `
struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

#inject <ignis/material/uniform-block>

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) worldPosition: vec3<f32>,
	@location(1) worldNormal: vec3<f32>,
	@location(2) uv0: vec2<f32>,
	@location(3) worldTangent: vec4<f32>,
}

fn roomLighting(faceNormal: vec3<f32>) -> f32 {
	let keyDir = normalize(vec3<f32>(-0.55, 0.75, 0.36));
	let keyPow = max(dot(faceNormal, keyDir), 0.0);
	let fillDir = normalize(vec3<f32>(0.60, -0.30, 0.74));
	let fillPow = max(dot(faceNormal, fillDir), 0.0) * 0.35;
	let ambient = 0.18;
	return ambient + keyPow * 0.72 + fillPow;
}

fn edgeAO(u: f32, v: f32) -> f32 {
	let eu = min(u, 1.0 - u);
	let ev = min(v, 1.0 - v);
	return smoothstep(0.0, 0.10, min(eu, ev));
}

@fragment
fn fsMainSingle(input: VertexOutput) -> @location(0) vec4<f32> {
	let viewDirWorld = normalize(input.worldPosition - frame.cameraPosition.xyz);

	let N = normalize(input.worldNormal);
	let T = normalize(input.worldTangent.xyz);
	let B = normalize(cross(N, T) * input.worldTangent.w);

	let aspect = ignisShaderUniforms.roomAspect;
	let rdRaw = vec3<f32>(
		dot(viewDirWorld, T),
		dot(viewDirWorld, B),
		-dot(viewDirWorld, N)
	);
	let rdAspect = vec3<f32>(rdRaw.x / aspect, rdRaw.y, rdRaw.z);

	if (rdAspect.z <= 0.0) {
		return vec4<f32>(0.04, 0.04, 0.04, 1.0);
	}

	let tiling = ignisShaderUniforms.roomTiling;
	let uvsScaled = input.uv0 * tiling;
	let ro = vec3<f32>(fract(uvsScaled.x), fract(uvsScaled.y), 0.0);
	let depth = ignisShaderUniforms.roomDepth;

	let EPS = 1e-5;
	let rdx = select(
		rdAspect.x,
		sign(rdAspect.x + 0.5) * EPS,
		abs(rdAspect.x) < EPS
	);
	let rdy = select(
		rdAspect.y,
		sign(rdAspect.y + 0.5) * EPS,
		abs(rdAspect.y) < EPS
	);
	let rdz = rdAspect.z;

	let tx = (select(0.0, 1.0, rdx > 0.0) - ro.x) / rdx;
	let ty = (select(0.0, 1.0, rdy > 0.0) - ro.y) / rdy;
	let tz = (depth - ro.z) / rdz;

	let t = min(tx, min(ty, tz));
	if (t < 0.0) {
		return vec4<f32>(0.04, 0.04, 0.04, 1.0);
	}
	let hit = ro + t * rdAspect;

	let hitX = t < (tx + EPS) && tx <= (ty + EPS) && tx <= (tz + EPS);
	let hitY = !hitX && t < (ty + EPS) && ty <= (tz + EPS);
	let depthPct = clamp(hit.z / depth, 0.0, 1.0);

	var baseColor = vec3<f32>(0.0);
	var faceNormal = vec3<f32>(0.0);
	var faceU = 0.0;
	var faceV = 0.0;

	if (hitX) {
		let wallDepthU = hit.z / depth;
		if (rdx > 0.0) {
			faceNormal = vec3<f32>(-1.0, 0.0, 0.0);
			faceU = 1.0 - wallDepthU;
			baseColor = vec3<f32>(0.82, 0.26, 0.26);
		} else {
			faceNormal = vec3<f32>(1.0, 0.0, 0.0);
			faceU = wallDepthU;
			baseColor = vec3<f32>(0.25, 0.80, 0.42);
		}
		faceV = hit.y;
		let baseboard = smoothstep(0.06, 0.05, hit.y);
		baseColor = mix(baseColor, vec3<f32>(0.88, 0.86, 0.82), baseboard);
	} else if (hitY) {
		faceU = hit.x;
		faceV = hit.z / depth;
		if (rdy > 0.0) {
			faceNormal = vec3<f32>(0.0, -1.0, 0.0);
			baseColor = vec3<f32>(0.93, 0.93, 0.90);
			let beam = smoothstep(0.03, 0.0, abs(fract(hit.x * 2.0) - 0.5) - 0.47);
			baseColor = mix(baseColor, vec3<f32>(0.70, 0.68, 0.62), beam * 0.5);
		} else {
			faceNormal = vec3<f32>(0.0, 1.0, 0.0);
			baseColor = vec3<f32>(0.52, 0.40, 0.28);
			let plankV = fract(hit.x * 6.0);
			let plankLine = smoothstep(0.04, 0.0, abs(plankV - 0.5) - 0.46);
			let crossV = fract(hit.z / depth * 3.0);
			let crossLine = smoothstep(0.04, 0.0, abs(crossV - 0.5) - 0.46);
			baseColor = mix(
				baseColor,
				vec3<f32>(0.28, 0.20, 0.12),
				max(plankLine, crossLine) * 0.6
			);
		}
	} else {
		faceNormal = vec3<f32>(0.0, 0.0, 1.0);
		faceU = hit.x;
		faceV = hit.y;
		baseColor = vec3<f32>(0.91, 0.87, 0.80);

		let stripe = smoothstep(0.03, 0.0, abs(fract(hit.x * 5.0) - 0.5) - 0.46);
		baseColor = mix(baseColor, vec3<f32>(0.78, 0.72, 0.65), stripe * 0.4);

		let lx = hit.x;
		let ly = hit.y;
		if (lx > 0.27 && lx < 0.73 && ly > 0.30 && ly < 0.78) {
			baseColor = vec3<f32>(0.18, 0.12, 0.06);
			if (lx > 0.31 && lx < 0.69 && ly > 0.34 && ly < 0.74) {
				let gradY = (ly - 0.34) / 0.40;
				let sky = mix(
					vec3<f32>(0.96, 0.40, 0.10),
					vec3<f32>(0.12, 0.18, 0.68),
					gradY
				);
				let sunDist = distance(vec2<f32>(lx, ly), vec2<f32>(0.50, 0.42));
				let sun = smoothstep(0.07, 0.04, sunDist);
				baseColor = mix(sky, vec3<f32>(1.0, 0.90, 0.60), sun);
			}
		}
	}

	let light = roomLighting(faceNormal);
	var color = baseColor * light;

	let depthAO = mix(1.0, 0.28, pow(depthPct, 1.4));
	color = color * depthAO;

	let cornerOcc = edgeAO(faceU, faceV);
	color = color * mix(0.38, 1.0, cornerOcc);

	color = color / (color + vec3<f32>(1.0));
	color = pow(color, vec3<f32>(1.0 / 2.2));

	return vec4<f32>(color, 1.0);
}
`;
