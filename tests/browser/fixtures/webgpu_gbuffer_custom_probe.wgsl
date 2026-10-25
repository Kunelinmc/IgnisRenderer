// These layouts match the engine frame-camera and object uniform ABI.
struct ProbeFrame {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	inverseView: mat4x4<f32>,
	inverseProjection: mat4x4<f32>,
	jitterNdc: vec2<f32>,
	prevJitterNdc: vec2<f32>,
}
struct ProbeObject {
	modelMatrix: mat4x4<f32>,
	prevModelMatrix: mat4x4<f32>,
	normalMatrix: mat4x4<f32>,
	instanceData: vec4<f32>,
}
@group(0) @binding(0) var<uniform> probeFrame: ProbeFrame;
@group(1) @binding(0) var<uniform> probeObject: ProbeObject;

struct ProbeVertex {
	@location(0) position: vec3<f32>,
	@location(2) normal: vec3<f32>,
}
struct Input {
	@invariant @builtin(position) position: vec4<f32>,
	@location(0) normal: vec3<f32>,
	@location(1) linearDepth: f32,
}
struct Output {
	@location(0) albedo: vec4<f32>,
	@location(1) normal: vec4<f32>,
	@location(2) emissive: vec4<f32>,
	@location(3) motion: vec4<f32>,
	@location(4) specular: vec4<f32>,
	@location(5) coat: vec4<f32>,
	@location(6) sheen: vec4<f32>,
	@builtin(frag_depth) depth: f32,
	@builtin(sample_mask) mask: u32,
}
struct ExtensionOutput {
	@location(0) materialExt0: vec4<f32>,
	@location(1) materialExt3: vec4<u32>,
	@builtin(frag_depth) depth: f32,
	@builtin(sample_mask) mask: u32,
}
struct SurfaceOutput {
	base: Output,
	extensions: ExtensionOutput,
}

@vertex fn vsProbe(input: ProbeVertex) -> Input {
	let world = probeObject.modelMatrix * vec4<f32>(input.position, 1.0);
	var clip = probeFrame.viewProjection * world;
	clip.x += probeFrame.jitterNdc.x * clip.w;
	clip.y += probeFrame.jitterNdc.y * clip.w;
	// Engine cameras use OpenGL clip depth; WebGPU raster depth is [0, 1].
	clip.z = clip.z * 0.5 + clip.w * 0.5;
	let normal = normalize((probeObject.normalMatrix * vec4<f32>(input.normal, 0.0)).xyz);
	let linearDepth = -dot(world.xyz - probeFrame.inverseView[3].xyz,
		probeFrame.inverseView[2].xyz);
	return Input(clip, normal, linearDepth);
}

fn evaluateProbe(input: Input) -> SurfaceOutput {
	// Generated ShaderMaterial uniforms drive identical coverage in both replays.
	if (input.position.x < ignisShaderUniforms.controls.x) { discard; }
	let depth = clamp(input.position.z + ignisShaderUniforms.controls.y, 0.0, 1.0);
	let mask = select(0u, 0xffffffffu, ignisShaderUniforms.controls.z > 0.5);
	let encodedNormal = normalize(input.normal).xy * 0.5 + vec2<f32>(0.5);
	let base = Output(ignisShaderUniforms.color,
		vec4<f32>(encodedNormal, 0.4, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0),
		// The exactly representable material word 2 selects the Unlit resolve.
		vec4<f32>(0.0, 0.0, input.linearDepth, 2.0), vec4<f32>(1.0),
		vec4<f32>(0.0), vec4<f32>(0.0), depth, mask);
	let extensions = ExtensionOutput(ignisShaderUniforms.ext0,
		ignisShaderUniforms.ext3, depth, mask);
	return SurfaceOutput(base, extensions);
}

@fragment fn fsProbeBase(input: Input) -> Output {
	return evaluateProbe(input).base;
}

@fragment fn fsProbeExtensions(input: Input) -> ExtensionOutput {
	return evaluateProbe(input).extensions;
}
