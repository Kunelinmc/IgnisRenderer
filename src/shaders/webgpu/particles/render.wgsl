#import <ignis/webgpu/constants>
#import <ignis/postprocess/fog>
struct FrameCameraUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
	ambientColor: vec4<f32>,
	lightCounts: vec4<f32>,
	options: vec4<f32>,
	environmentOptionsA: vec4<f32>,
	environmentOptionsB: vec4<f32>,
	taaJitterCurrentPrev: vec4<f32>,
}

struct FrameShadowUniforms {
	directionalShadows: array<ShadowData, __WEBGPU_MAX_DIRECTIONAL_LIGHTS__>,
	spotShadows: array<ShadowData, __WEBGPU_MAX_SPOT_LIGHTS__>,
}

struct ParticleVertexInput {
	@location(0) quadPosition: vec2<f32>,
	@location(1) quadUV: vec2<f32>,
	@location(2) instancePositionSize: vec4<f32>,
	@location(3) instanceColor: vec4<f32>,
	@location(4) instanceUVRect: vec4<f32>,
	@location(5) instanceRotation: f32,
	@location(6) instanceReceiveShadow: f32,
}

struct ParticleVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
	@location(1) color: vec4<f32>,
	@location(2) worldPosition: vec3<f32>,
	@location(3) receiveShadow: f32,
	@location(4) localUV: vec2<f32>,
}

struct ParticleUVTransform {
	transformA: vec4<f32>, // xy: repeat, zw: offset
	transformB: vec4<f32>, // x: cos(rotation), y: sin(rotation)
}

struct ParticleOITOutput {
	@location(0) accum: vec4<f32>,
	@location(1) reveal: vec4<f32>,
}

struct FogUniforms {
	fogParams0: vec4<f32>,
	fogParams1: vec4<f32>,
}

struct ParticleShadowVolumeBuffer {
	data: array<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameCameraUniforms;
@group(0) @binding(1) var shadowAtlas: texture_depth_2d;
@group(0) @binding(2) var envSpecularTexture: texture_2d<f32>;
@group(0) @binding(3) var envSpecularSampler: sampler;
@group(0) @binding(6) var<uniform> fog: FogUniforms;
@group(0) @binding(7) var<storage, read> particleShadowVolumes:
	ParticleShadowVolumeBuffer;
@group(0) @binding(8) var shadowTransmittanceAtlas: texture_2d<f32>;
@group(0) @binding(11) var pagedShadowPageTable: texture_2d<u32>;
@group(0) @binding(12) var pagedShadowPhysicalDepth: texture_depth_2d;
@group(0) @binding(13) var shadowComparisonSampler: sampler_comparison;
@group(0) @binding(15) var<uniform> frameShadows: FrameShadowUniforms;

@group(1) @binding(0) var particleTexture: texture_2d<f32>;
@group(1) @binding(1) var particleSampler: sampler;
@group(1) @binding(2) var<uniform> particleUVTransform: ParticleUVTransform;

fn resolveParticleOITWeight(alpha: f32, viewDepth: f32) -> f32 {
	let clampedAlpha = clamp(alpha, 0.0, 1.0);
	let normalizedDepth = clamp(viewDepth / 400.0, 0.0, 1.0);
	let depthWeight = clamp(1.0 - normalizedDepth, 0.05, 1.0);
	let alphaWeight = max(clampedAlpha * 8.0 + 0.01, 0.01);
	let weight = alphaWeight * alphaWeight * alphaWeight * depthWeight;
	return clamp(weight, 1e-2, 3e3);
}

@vertex
fn vsMain(input: ParticleVertexInput) -> ParticleVertexOutput {
	var output: ParticleVertexOutput;
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let c = cos(input.instanceRotation);
	let s = sin(input.instanceRotation);
	let rotated = vec2<f32>(
		input.quadPosition.x * c - input.quadPosition.y * s,
		input.quadPosition.x * s + input.quadPosition.y * c
	);
	let worldPosition = input.instancePositionSize.xyz +
		(right * rotated.x + up * rotated.y) * input.instancePositionSize.w;

	var clipPosition = frame.viewProjection * vec4<f32>(worldPosition, 1.0);
	let currJitter = frame.taaJitterCurrentPrev.xy * clipPosition.w;
	clipPosition = vec4<f32>(
		clipPosition.x + currJitter.x,
		clipPosition.y + currJitter.y,
		clipPosition.z,
		clipPosition.w
	);
	clipPosition.z = clipPosition.z * 0.5 + clipPosition.w * 0.5;

	output.position = clipPosition;
	output.uv = vec2<f32>(
		mix(input.instanceUVRect.x, input.instanceUVRect.z, input.quadUV.x),
		mix(input.instanceUVRect.y, input.instanceUVRect.w, input.quadUV.y)
	);
	output.color = input.instanceColor;
	output.worldPosition = worldPosition;
	output.receiveShadow = input.instanceReceiveShadow;
	output.localUV = input.quadUV;
	return output;
}

fn resolveParticleDirectionalCascade(shadow: ShadowData, linearDepth: f32) -> u32 {
	let cascadeCount = u32(clamp(floor(shadow.paramsC.z + 0.5), 1.0, 4.0));
	if (cascadeCount <= 1u || shadow.paramsC.y < 0.5) {
		return 0u;
	}

	var selected = cascadeCount - 1u;
	for (var i = 0u; i < 4u; i = i + 1u) {
		if (i >= cascadeCount) {
			break;
		}
		if (linearDepth <= shadow.cascadeSplits[i].y) {
			selected = i;
			break;
		}
	}
	return selected;
}

fn sampleParticlePagedShadow(
	shadow: ShadowData,
	uv: vec2<f32>,
	currentDepth: f32,
	cascadeIndex: u32
) -> vec3<f32> {
	let pageGridSize = max(u32(floor(shadow.paramsE.z + 0.5)), 1u);
	let pageSize = max(i32(floor(shadow.paramsF.z + 0.5)), 1);
	let physicalGridSize = max(u32(floor(shadow.paramsF.y + 0.5)), 1u);
	let pageTableBase = u32(max(floor(shadow.paramsE.y + 0.5), 0.0));
	let cascadeStride = max(
		u32(floor(shadow.paramsF.w + 0.5)),
		pageGridSize * pageGridSize
	);
	let pageCoord = vec2<u32>(
		u32(clamp(floor(uv.x * f32(pageGridSize)), 0.0, f32(pageGridSize - 1u))),
		u32(clamp(floor(uv.y * f32(pageGridSize)), 0.0, f32(pageGridSize - 1u)))
	);
	let pageTableWidth = max(textureDimensions(pagedShadowPageTable).x, 1u);
	let pageTableIndex = pageTableBase + cascadeIndex * cascadeStride +
		pageCoord.y * pageGridSize + pageCoord.x;
	let physicalPageIndex = textureLoad(
		pagedShadowPageTable,
		vec2<i32>(
			i32(pageTableIndex % pageTableWidth),
			i32(pageTableIndex / pageTableWidth)
		),
		0
	).x;
	if (
		physicalPageIndex == 0xffffffffu ||
		physicalPageIndex >= physicalGridSize * physicalGridSize
	) {
		return vec3<f32>(1.0);
	}

	let physicalPageCoord = vec2<i32>(
		i32(physicalPageIndex % physicalGridSize),
		i32(physicalPageIndex / physicalGridSize)
	);
	let localUv = fract(uv * vec2<f32>(f32(pageGridSize)));
	let atlasPosition = vec2<f32>(physicalPageCoord * pageSize) +
		localUv * vec2<f32>(f32(pageSize - 1)) + vec2<f32>(0.5);
	let dimensions = vec2<f32>(textureDimensions(pagedShadowPhysicalDepth));
	let visibility = textureSampleCompareLevel(
		pagedShadowPhysicalDepth,
		shadowComparisonSampler,
		atlasPosition / max(dimensions, vec2<f32>(1.0)),
		currentDepth - max(shadow.paramsA.y, 0.0)
	);
	let strength = clamp(shadow.paramsB.y, 0.0, 1.0);
	return vec3<f32>(1.0 - strength + strength * visibility);
}

fn sampleParticleDirectionalCascade(
	index: u32,
	shadow: ShadowData,
	worldPosition: vec3<f32>,
	cascadeIndex: u32
) -> vec3<f32> {
	let isCSM = shadow.paramsC.y > 0.5 && shadow.paramsC.z > 1.5;
	let split = shadow.cascadeSplits[cascadeIndex];
	var shadowMatrix = shadow.viewProjection;
	if (isCSM) {
		shadowMatrix = shadow.cascadeViewProjections[cascadeIndex];
	}
	let lightClip = shadowMatrix * vec4<f32>(worldPosition, 1.0);
	if (lightClip.w <= 1e-6) {
		return vec3<f32>(1.0);
	}

	let ndc = lightClip.xyz / lightClip.w;
	let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
	let currentDepth = ndc.z * 0.5 + 0.5;
	if (
		any(uv < vec2<f32>(0.0)) ||
		any(uv > vec2<f32>(1.0)) ||
		currentDepth < 0.0 ||
		currentDepth > 1.0
	) {
		return vec3<f32>(1.0);
	}

	if (shadow.paramsE.x > 0.5) {
		return sampleParticlePagedShadow(shadow, uv, currentDepth, cascadeIndex) *
			sampleParticleShadowVolumeTransmittance(
				0u,
				index,
				cascadeIndex,
				worldPosition
			);
	}

	let requestedSize = max(i32(shadow.paramsB.z + 0.5), 1);
	let tileSize = max(i32(shadow.paramsB.w + 0.5), requestedSize);
	let localTileSpan = select(1, 2, isCSM);
	let subTileSize = max(tileSize / localTileSpan, 1);
	let shadowSize = max(min(requestedSize, subTileSize), 1);
	let atlasDimensions = textureDimensions(shadowAtlas);
	let atlasColumns = max(i32(atlasDimensions.x) / tileSize, 1);
	let tileOffset = vec2<i32>(
		i32(index % u32(atlasColumns)) * tileSize,
		i32(index / u32(atlasColumns)) * tileSize
	) + vec2<i32>(
		select(0, i32(clamp(floor(split.z + 0.5), 0.0, 1.0)), isCSM) * subTileSize,
		select(0, i32(clamp(floor(split.w + 0.5), 0.0, 1.0)), isCSM) * subTileSize
	);
	let texelPosition = uv * vec2<f32>(f32(shadowSize - 1));
	let atlasPosition = vec2<f32>(tileOffset) + texelPosition + vec2<f32>(0.5);
	let visibility = textureSampleCompareLevel(
		shadowAtlas,
		shadowComparisonSampler,
		atlasPosition / max(vec2<f32>(atlasDimensions), vec2<f32>(1.0)),
		currentDepth - max(shadow.paramsA.y, 0.0)
	);
	let localCoord = vec2<i32>(round(texelPosition));
	let transmittance = textureLoad(
		shadowTransmittanceAtlas,
		tileOffset + clamp(
			localCoord,
			vec2<i32>(0),
			vec2<i32>(shadowSize - 1)
		),
		0
	).rgb;
	let strength = clamp(shadow.paramsB.y, 0.0, 1.0);
	return (vec3<f32>(1.0 - strength) + strength * visibility * transmittance) *
		sampleParticleShadowVolumeTransmittance(
			0u,
			index,
			cascadeIndex,
			worldPosition
		);
}

fn sampleDirectionalShadowVisibility(worldPosition: vec3<f32>) -> vec3<f32> {
	if (frame.options.z < 0.5) {
		return vec3<f32>(1.0);
	}

	let directionalCount = min(
		u32(max(frame.lightCounts.x + 0.5, 0.0)),
		u32(__WEBGPU_MAX_DIRECTIONAL_LIGHTS__)
	);
	let linearDepth = dot(
		frame.cameraPosition.xyz - worldPosition,
		frame.environmentBasisBackward.xyz
	);
	var visibility = vec3<f32>(1.0);
	for (var index = 0u; index < directionalCount; index = index + 1u) {
		let shadow = frameShadows.directionalShadows[index];
		if (shadow.paramsA.x < 0.5) {
			continue;
		}
		let cascadeIndex = resolveParticleDirectionalCascade(shadow, linearDepth);
		visibility = min(
			visibility,
			sampleParticleDirectionalCascade(
				index,
				shadow,
				worldPosition,
				cascadeIndex
			)
		);
	}
	return visibility;
}

fn sampleParticleShadowVolumeTransmittance(
	shadowType: u32,
	index: u32,
	cascadeIndex: u32,
	worldPosition: vec3<f32>
) -> f32 {
	if (shadowType != 0u || index != 0u || cascadeIndex >= 4u) {
		return 1.0;
	}

	let metaBase = cascadeIndex * 24u;
	let bufferLength = arrayLength(&particleShadowVolumes.data);
	if (metaBase + 20u >= bufferLength) {
		return 1.0;
	}
	if (particleShadowVolumes.data[metaBase + 16u] < 0.5) {
		return 1.0;
	}

	let width = u32(max(particleShadowVolumes.data[metaBase + 17u], 1.0));
	let height = u32(max(particleShadowVolumes.data[metaBase + 18u], 1.0));
	let depth = u32(max(particleShadowVolumes.data[metaBase + 19u], 1.0));
	let densityOffset = u32(max(particleShadowVolumes.data[metaBase + 20u], 0.0));
	let viewProjection = mat4x4<f32>(
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 0u],
			particleShadowVolumes.data[metaBase + 1u],
			particleShadowVolumes.data[metaBase + 2u],
			particleShadowVolumes.data[metaBase + 3u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 4u],
			particleShadowVolumes.data[metaBase + 5u],
			particleShadowVolumes.data[metaBase + 6u],
			particleShadowVolumes.data[metaBase + 7u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 8u],
			particleShadowVolumes.data[metaBase + 9u],
			particleShadowVolumes.data[metaBase + 10u],
			particleShadowVolumes.data[metaBase + 11u]
		),
		vec4<f32>(
			particleShadowVolumes.data[metaBase + 12u],
			particleShadowVolumes.data[metaBase + 13u],
			particleShadowVolumes.data[metaBase + 14u],
			particleShadowVolumes.data[metaBase + 15u]
		)
	);
	let clip = viewProjection * vec4<f32>(worldPosition, 1.0);
	if (clip.w <= 1e-6) {
		return 1.0;
	}
	let ndc = clip.xyz / clip.w;
	if (
		ndc.x < -1.0 || ndc.x > 1.0 ||
		ndc.y < -1.0 || ndc.y > 1.0 ||
		ndc.z < -1.0 || ndc.z > 1.0
	) {
		return 1.0;
	}

	let x = u32(clamp(
		round((ndc.x * 0.5 + 0.5) * f32(width - 1u)),
		0.0,
		f32(width - 1u)
	));
	let y = u32(clamp(
		round((0.5 - ndc.y * 0.5) * f32(height - 1u)),
		0.0,
		f32(height - 1u)
	));
	let zMax = u32(clamp(
		round((ndc.z * 0.5 + 0.5) * f32(depth - 1u)),
		0.0,
		f32(depth - 1u)
	));

	var opticalDepth = 0.0;
	var z = 0u;
	loop {
		if (z > zMax) {
			break;
		}
		let densityIndex = densityOffset + z * width * height + y * width + x;
		if (densityIndex < bufferLength) {
			opticalDepth += particleShadowVolumes.data[densityIndex];
		}
		z = z + 1u;
	}

	return exp(-max(opticalDepth / max(f32(depth), 1.0), 0.0));
}

fn shadeParticle(input: ParticleVertexOutput) -> vec4<f32> {
	// Procedural soft radial falloff (circle mask)
	let dist = distance(input.localUV, vec2<f32>(0.5, 0.5));
	let radialMask = 1.0 - smoothstep(0.4, 0.5, dist);
	
	let scaledUV = vec2<f32>(
		input.uv.x * particleUVTransform.transformA.x,
		input.uv.y * particleUVTransform.transformA.y
	);
	let rotatedUV = vec2<f32>(
		scaledUV.x * particleUVTransform.transformB.x -
			scaledUV.y * particleUVTransform.transformB.y,
		scaledUV.x * particleUVTransform.transformB.y +
			scaledUV.y * particleUVTransform.transformB.x
	);
	let sampled = textureSample(
		particleTexture,
		particleSampler,
		rotatedUV + particleUVTransform.transformA.zw
	);
	
	var color = sampled * input.color;
	color.a = color.a * radialMask;
	
	if (color.a <= 0.001) {
		discard;
	}
	if (input.receiveShadow > 0.5) {
		let visibility = sampleDirectionalShadowVisibility(input.worldPosition);
		color = vec4<f32>(color.rgb * visibility, color.a);
	}

	let viewDepth = length(frame.cameraPosition.xyz - input.worldPosition);
	let fogMode = i32(floor(fog.fogParams0.x + 0.5));
	let fogFactor = ignisComputeFogFactor(
		fogMode,
		max(viewDepth, 0.0),
		fog.fogParams0.y,
		fog.fogParams0.z,
		fog.fogParams0.w,
		fog.fogParams1.w
	);
	color = vec4<f32>(
		max(mix(color.rgb, fog.fogParams1.rgb, fogFactor), vec3<f32>(0.0)),
		color.a
	);
	return color;
}

@fragment
fn fsMain(input: ParticleVertexOutput) -> @location(0) vec4<f32> {
	return shadeParticle(input);
}

@fragment
fn fsMainOIT(input: ParticleVertexOutput) -> ParticleOITOutput {
	let color = shadeParticle(input);
	let alpha = clamp(color.a, 0.0, 1.0);
	let viewDepth = length(frame.cameraPosition.xyz - input.worldPosition);
	let weight = resolveParticleOITWeight(alpha, viewDepth);
	var output: ParticleOITOutput;
	output.accum = vec4<f32>(color.rgb * alpha, alpha) * weight;
	output.reveal = vec4<f32>(alpha, alpha, alpha, alpha);
	return output;
}
