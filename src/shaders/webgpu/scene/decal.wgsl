const EPSILON: f32 = 1e-6;
const TEX_BASE_COLOR: u32 = 0u;
const TEX_METALLIC_ROUGHNESS: u32 = 1u;
const TEX_NORMAL: u32 = 2u;
const TEX_EMISSIVE: u32 = 3u;
const TEX_OCCLUSION: u32 = 4u;
const TEX_SPECULAR: u32 = 5u;
const TEX_SPECULAR_COLOR: u32 = 6u;
const TEX_CLEARCOAT: u32 = 7u;
const TEX_CLEARCOAT_ROUGHNESS: u32 = 8u;
const TEX_CLEARCOAT_NORMAL: u32 = 9u;
const TEX_SHEEN_COLOR: u32 = 10u;
const TEX_SHEEN_ROUGHNESS: u32 = 11u;
const TEX_TRANSMISSION: u32 = 12u;
const TEX_THICKNESS: u32 = 13u;
const TEX_IRIDESCENCE: u32 = 14u;
const TEX_IRIDESCENCE_THICKNESS: u32 = 15u;

const MODE_DISABLED: u32 = 0u;
const MODE_LERP: u32 = 1u;
const MODE_REPLACE: u32 = 2u;
const MODE_MULTIPLY: u32 = 3u;
const MODE_ADD: u32 = 4u;
const MODE_NORMAL: u32 = 5u;

struct FrameUniforms {
	viewProjection: mat4x4<f32>,
	prevViewProjection: mat4x4<f32>,
	cameraPosition: vec4<f32>,
	environmentBasisRight: vec4<f32>,
	environmentBasisUp: vec4<f32>,
	environmentBasisBackward: vec4<f32>,
}

struct DecalUniforms {
	worldToLocal: mat4x4<f32>,
	localToWorld: mat4x4<f32>,
	projectorParams: vec4<f32>,
	baseColorFactor: vec4<f32>,
	emissiveFactor: vec4<f32>,
	surfaceParams0: vec4<f32>,
	surfaceParams1: vec4<f32>,
	surfaceParams2: vec4<f32>,
	surfaceParams3: vec4<f32>,
	specularColorFactor: vec4<f32>,
	phongAmbientShininess: vec4<f32>,
	phongSpecularShading: vec4<f32>,
	sheenColorClearcoatNormalScale: vec4<f32>,
	attenuationColor: vec4<f32>,
	anisotropyParams: vec4<f32>,
	anisotropyTextureTransformA: vec4<f32>,
	anisotropyTextureTransformB: vec4<f32>,
	materialFlags: vec4<f32>,
	textureTransformA: array<vec4<f32>, 16>,
	textureTransformB: array<vec4<f32>, 16>,
	channelModes: array<vec4<f32>, 5>,
}

struct VSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

struct GBufferOutput {
	@location(0) gAlbedoAlpha: vec4<f32>,
	@location(1) gNormalRoughMetal: vec4<f32>,
	@location(2) gEmissiveOcclusion: vec4<f32>,
	@location(3) gMotionDepth: vec4<f32>,
	@location(4) gSpecular: vec4<f32>,
	@location(5) gCoatSheen: vec4<f32>,
	@location(6) gSheenReflectance: vec4<f32>,
}

struct DecalBatchParams {
	rect: vec4<u32>,
	tileInfo: vec4<u32>,
}

struct DecalEvaluation {
	applied: u32,
	gAlbedoAlpha: vec4<f32>,
	gNormalRoughMetal: vec4<f32>,
	gEmissiveOcclusion: vec4<f32>,
	gMotionDepth: vec4<f32>,
	gSpecular: vec4<f32>,
	gCoatSheen: vec4<f32>,
	gSheenReflectance: vec4<f32>,
	gMaterialExt0: vec4<f32>,
	gMaterialExt1: vec4<f32>,
	gMaterialExt2: vec4<f32>,
	gMaterialExt3: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

@group(1) @binding(0) var gAlbedoAlphaIn: texture_2d<f32>;
@group(1) @binding(1) var gNormalRoughMetalIn: texture_2d<f32>;
@group(1) @binding(2) var gEmissiveOcclusionIn: texture_2d<f32>;
@group(1) @binding(3) var gMotionDepthIn: texture_2d<f32>;
@group(1) @binding(4) var gSpecularIn: texture_2d<f32>;
@group(1) @binding(5) var gCoatSheenIn: texture_2d<f32>;
@group(1) @binding(6) var gSheenReflectanceIn: texture_2d<f32>;
@group(1) @binding(7) var gMaterialExt0In: texture_2d<f32>;
@group(1) @binding(8) var gMaterialExt1In: texture_2d<f32>;
@group(1) @binding(9) var gMaterialExt2In: texture_2d<f32>;
@group(1) @binding(10) var gMaterialExt3In: texture_2d<f32>;

@group(2) @binding(0) var<uniform> decal: DecalUniforms;
@group(2) @binding(1) var baseColorTexture: texture_2d<f32>;
@group(2) @binding(2) var baseColorSampler: sampler;
@group(2) @binding(3) var metallicRoughnessTexture: texture_2d<f32>;
@group(2) @binding(4) var metallicRoughnessSampler: sampler;
@group(2) @binding(5) var normalTexture: texture_2d<f32>;
@group(2) @binding(6) var normalSampler: sampler;
@group(2) @binding(7) var emissiveTexture: texture_2d<f32>;
@group(2) @binding(8) var emissiveSampler: sampler;
@group(2) @binding(9) var occlusionTexture: texture_2d<f32>;
@group(2) @binding(10) var occlusionSampler: sampler;
@group(2) @binding(11) var specularTexture: texture_2d<f32>;
@group(2) @binding(12) var specularSampler: sampler;
@group(2) @binding(13) var specularColorTexture: texture_2d<f32>;
@group(2) @binding(14) var specularColorSampler: sampler;
@group(2) @binding(15) var clearcoatTexture: texture_2d<f32>;
@group(2) @binding(16) var clearcoatSampler: sampler;
@group(2) @binding(17) var clearcoatRoughnessTexture: texture_2d<f32>;
@group(2) @binding(18) var clearcoatRoughnessSampler: sampler;
@group(2) @binding(19) var clearcoatNormalTexture: texture_2d<f32>;
@group(2) @binding(20) var clearcoatNormalSampler: sampler;
@group(2) @binding(21) var sheenColorTexture: texture_2d<f32>;
@group(2) @binding(22) var sheenColorSampler: sampler;
@group(2) @binding(23) var sheenRoughnessTexture: texture_2d<f32>;
@group(2) @binding(24) var sheenRoughnessSampler: sampler;
@group(2) @binding(25) var transmissionTexture: texture_2d<f32>;
@group(2) @binding(26) var transmissionSampler: sampler;
@group(2) @binding(27) var thicknessTexture: texture_2d<f32>;
@group(2) @binding(29) var iridescenceTexture: texture_2d<f32>;
@group(2) @binding(31) var iridescenceThicknessTexture: texture_2d<f32>;
@group(2) @binding(37) var anisotropyTexture: texture_2d<f32>;

@group(3) @binding(0) var<uniform> batch: DecalBatchParams;
@group(3) @binding(1) var<storage, read> batchDecals: array<DecalUniforms>;
@group(3) @binding(2) var<storage, read> batchTileHeaders: array<vec4<u32>>;
@group(3) @binding(3) var<storage, read> batchTileDecalIndices: array<u32>;
@group(3) @binding(4) var gAlbedoAlphaBatchOut: texture_storage_2d<rgba8unorm, write>;
@group(3) @binding(5) var gNormalRoughMetalBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(6) var gEmissiveOcclusionBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var gMotionDepthBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(8) var gSpecularBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(9) var gCoatSheenBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(10) var gSheenReflectanceBatchOut: texture_storage_2d<rgba16float, write>;
@group(3) @binding(11) var gMaterialExt0Out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(12) var gMaterialExt1Out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(13) var gMaterialExt2Out: texture_storage_2d<rgba16float, write>;
@group(3) @binding(14) var gMaterialExt3Out: texture_storage_2d<rgba16float, write>;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
	let positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	let position = positions[vertexIndex];
	var output: VSOut;
	output.position = vec4<f32>(position, 0.0, 1.0);
	output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
	return output;
}

fn saturate(value: f32) -> f32 {
	return clamp(value, 0.0, 1.0);
}

fn safeNormalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(value);
	return select(fallback, value / max(len, EPSILON), len > EPSILON);
}

fn srgbChannelToLinear(value: f32) -> f32 {
	return select(
		pow(max(value, 0.0), 2.2),
		value,
		value <= 0.0
	);
}

fn srgbToLinear(value: vec3<f32>) -> vec3<f32> {
	return vec3<f32>(
		srgbChannelToLinear(value.r),
		srgbChannelToLinear(value.g),
		srgbChannelToLinear(value.b)
	);
}

fn octahedralWrap(v: vec2<f32>) -> vec2<f32> {
	return (vec2<f32>(1.0) - abs(v.yx)) * select(
		vec2<f32>(-1.0),
		vec2<f32>(1.0),
		v.xy >= vec2<f32>(0.0)
	);
}

fn encodeOctahedralNormal(normal: vec3<f32>) -> vec2<f32> {
	let n = safeNormalize(normal, vec3<f32>(0.0, 0.0, 1.0));
	let denom = max(abs(n.x) + abs(n.y) + abs(n.z), EPSILON);
	var oct = n.xy / denom;
	if (n.z < 0.0) {
		oct = octahedralWrap(oct);
	}
	return oct * 0.5 + vec2<f32>(0.5);
}

fn decodeOctahedralNormal(encoded: vec2<f32>) -> vec3<f32> {
	let oct = encoded * 2.0 - vec2<f32>(1.0);
	var n = vec3<f32>(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
	if (n.z < 0.0) {
		n = vec3<f32>(octahedralWrap(n.xy), n.z);
	}
	return safeNormalize(n, vec3<f32>(0.0, 0.0, 1.0));
}

fn encodeNormalForGBuffer(normal: vec3<f32>) -> vec2<f32> {
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let backward = frame.environmentBasisBackward.xyz;
	let viewNormal = vec3<f32>(
		dot(normal, right),
		dot(normal, up),
		dot(normal, backward)
	);
	return encodeOctahedralNormal(viewNormal);
}

fn decodeDeferredNormal(encoded: vec2<f32>) -> vec3<f32> {
	let viewNormal = decodeOctahedralNormal(encoded);
	return safeNormalize(
		frame.environmentBasisRight.xyz * viewNormal.x +
			frame.environmentBasisUp.xyz * viewNormal.y +
			frame.environmentBasisBackward.xyz * viewNormal.z,
		frame.environmentBasisBackward.xyz
	);
}

fn reconstructDeferredWorldPosition(uv: vec2<f32>, depth: f32) -> vec3<f32> {
	let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
	let right = frame.environmentBasisRight.xyz;
	let up = frame.environmentBasisUp.xyz;
	let backward = frame.environmentBasisBackward.xyz;
	if (frame.environmentBasisBackward.w > 0.5) {
		return frame.cameraPosition.xyz +
			right * ndc.x * frame.environmentBasisRight.w +
			up * ndc.y * frame.environmentBasisUp.w -
			backward * depth;
	}
	let cx = ndc.x * frame.environmentBasisUp.w * frame.environmentBasisRight.w * depth;
	let cy = ndc.y * frame.environmentBasisRight.w * depth;
	return frame.cameraPosition.xyz + right * cx + up * cy - backward * depth;
}

fn transformUV(slotIndex: u32, uv: vec2<f32>) -> vec2<f32> {
	let transformA = decal.textureTransformA[slotIndex];
	let transformB = decal.textureTransformB[slotIndex];
	var transformed = uv * transformA.zw;
	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		transformed = vec2<f32>(
			transformed.x * c - transformed.y * s,
			transformed.x * s + transformed.y * c
		);
	}
	return transformed + transformA.xy;
}

fn transformAnisotropyUV(uv: vec2<f32>) -> vec2<f32> {
	let transformA = decal.anisotropyTextureTransformA;
	let transformB = decal.anisotropyTextureTransformB;
	var transformed = uv * transformA.zw;
	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		transformed = vec2<f32>(
			transformed.x * c - transformed.y * s,
			transformed.x * s + transformed.y * c
		);
	}
	return transformed + transformA.xy;
}

fn sampleLinearTexture(
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv: vec2<f32>
) -> vec4<f32> {
	return textureSample(textureRef, samplerRef, transformUV(slotIndex, uv));
}

fn sampleColorTexture(
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv: vec2<f32>
) -> vec4<f32> {
	let sampled = sampleLinearTexture(textureRef, samplerRef, slotIndex, uv);
	let isLinear = decal.textureTransformB[slotIndex].z > 0.5;
	return select(vec4<f32>(srgbToLinear(sampled.rgb), sampled.a), sampled, isLinear);
}

fn transformLocalDirection(direction: vec3<f32>) -> vec3<f32> {
	return safeNormalize((decal.localToWorld * vec4<f32>(direction, 0.0)).xyz, direction);
}

fn getChannelMode(index: u32) -> u32 {
	let vectorIndex = index / 4u;
	let componentIndex = index % 4u;
	let values = decal.channelModes[vectorIndex];
	let value = select(
		select(values.x, values.y, componentIndex == 1u),
		select(values.z, values.w, componentIndex == 3u),
		componentIndex >= 2u
	);
	return u32(clamp(floor(value + 0.5), 0.0, 5.0));
}

fn blendScalar(oldValue: f32, newValue: f32, mode: u32, weight: f32) -> f32 {
	let w = saturate(weight);
	if (mode == MODE_DISABLED || w <= 0.0) {
		return oldValue;
	}
	if (mode == MODE_MULTIPLY) {
		return mix(oldValue, oldValue * newValue, w);
	}
	if (mode == MODE_ADD) {
		return oldValue + newValue * w;
	}
	return mix(oldValue, newValue, w);
}

fn blendVec3(oldValue: vec3<f32>, newValue: vec3<f32>, mode: u32, weight: f32) -> vec3<f32> {
	let w = saturate(weight);
	if (mode == MODE_DISABLED || w <= 0.0) {
		return oldValue;
	}
	if (mode == MODE_MULTIPLY) {
		return mix(oldValue, oldValue * newValue, vec3<f32>(w));
	}
	if (mode == MODE_ADD) {
		return oldValue + newValue * w;
	}
	return mix(oldValue, newValue, vec3<f32>(w));
}

fn blendNormal(oldNormal: vec3<f32>, decalNormal: vec3<f32>, mode: u32, weight: f32) -> vec3<f32> {
	let w = saturate(weight);
	if (mode == MODE_DISABLED || w <= 0.0) {
		return oldNormal;
	}
	if (mode == MODE_NORMAL || mode == MODE_REPLACE || mode == MODE_LERP) {
		return safeNormalize(mix(oldNormal, decalNormal, vec3<f32>(w)), oldNormal);
	}
	return oldNormal;
}

fn layerMatches(pixelLayerMask: f32, receiverLayerMask: f32) -> bool {
	let pixelMask = u32(max(0.0, floor(pixelLayerMask + 0.5)));
	let receiverMask = u32(max(0.0, floor(receiverLayerMask + 0.5)));
	return (pixelMask & receiverMask) != 0u;
}

fn projectorOpacity(localPosition: vec3<f32>, baseAlpha: f32) -> f32 {
	let halfExtent = vec3<f32>(0.5);
	let distanceToEdge = min(
		min(halfExtent.x - abs(localPosition.x), halfExtent.y - abs(localPosition.y)),
		halfExtent.z - abs(localPosition.z)
	);
	if (distanceToEdge < 0.0) {
		return 0.0;
	}
	let edgeFade = max(decal.projectorParams.y, 0.0);
	let fade = select(1.0, saturate(distanceToEdge / max(edgeFade, EPSILON)), edgeFade > 0.0);
	return saturate(decal.projectorParams.x * baseAlpha * fade);
}

@fragment
fn fsMain(input: VSOut) -> GBufferOutput {
	let coord = vec2<i32>(i32(input.position.x), i32(input.position.y));
	let albedoAlphaOld = textureLoad(gAlbedoAlphaIn, coord, 0);
	let normalRoughMetalOld = textureLoad(gNormalRoughMetalIn, coord, 0);
	let emissiveOcclusionOld = textureLoad(gEmissiveOcclusionIn, coord, 0);
	let motionDepthOld = textureLoad(gMotionDepthIn, coord, 0);
	let specularOld = textureLoad(gSpecularIn, coord, 0);
	let coatSheenOld = textureLoad(gCoatSheenIn, coord, 0);
	let sheenReflectanceOld = textureLoad(gSheenReflectanceIn, coord, 0);
	let materialExt0Old = textureLoad(gMaterialExt0In, coord, 0);
	let materialExt1Old = textureLoad(gMaterialExt1In, coord, 0);
	let materialExt2Old = textureLoad(gMaterialExt2In, coord, 0);
	let materialExt3Old = textureLoad(gMaterialExt3In, coord, 0);

	if (motionDepthOld.z <= 0.0 || !layerMatches(materialExt3Old.w, decal.projectorParams.z)) {
		discard;
	}

	let dimensions = vec2<f32>(textureDimensions(gAlbedoAlphaIn, 0));
	let screenUV = input.position.xy / max(dimensions, vec2<f32>(1.0));
	let worldPosition = reconstructDeferredWorldPosition(screenUV, motionDepthOld.z);
	let localPosition = (decal.worldToLocal * vec4<f32>(worldPosition, 1.0)).xyz;
	let projectorUV = localPosition.xy + vec2<f32>(0.5);
	let baseSample = sampleColorTexture(
		baseColorTexture,
		baseColorSampler,
		TEX_BASE_COLOR,
		projectorUV
	);
	let opacity = projectorOpacity(localPosition, baseSample.a);
	if (opacity <= 0.0) {
		discard;
	}

	let oldNormal = decodeDeferredNormal(normalRoughMetalOld.xy);
	let baseColor = clamp(decal.baseColorFactor.rgb * baseSample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
	let mrSample = sampleLinearTexture(
		metallicRoughnessTexture,
		metallicRoughnessSampler,
		TEX_METALLIC_ROUGHNESS,
		projectorUV
	);
	let normalSample = sampleLinearTexture(
		normalTexture,
		normalSampler,
		TEX_NORMAL,
		projectorUV
	).rgb;
	let emissiveSample = sampleColorTexture(
		emissiveTexture,
		emissiveSampler,
		TEX_EMISSIVE,
		projectorUV
	);
	let occlusionSample = sampleLinearTexture(
		occlusionTexture,
		occlusionSampler,
		TEX_OCCLUSION,
		projectorUV
	);
	let specularSample = sampleLinearTexture(
		specularTexture,
		specularSampler,
		TEX_SPECULAR,
		projectorUV
	);
	let specularColorSample = sampleColorTexture(
		specularColorTexture,
		specularColorSampler,
		TEX_SPECULAR_COLOR,
		projectorUV
	);
	let clearcoatSample = sampleLinearTexture(
		clearcoatTexture,
		clearcoatSampler,
		TEX_CLEARCOAT,
		projectorUV
	);
	let clearcoatRoughnessSample = sampleLinearTexture(
		clearcoatRoughnessTexture,
		clearcoatRoughnessSampler,
		TEX_CLEARCOAT_ROUGHNESS,
		projectorUV
	);
	let clearcoatNormalSample = sampleLinearTexture(
		clearcoatNormalTexture,
		clearcoatNormalSampler,
		TEX_CLEARCOAT_NORMAL,
		projectorUV
	).rgb;
	let sheenColorSample = sampleColorTexture(
		sheenColorTexture,
		sheenColorSampler,
		TEX_SHEEN_COLOR,
		projectorUV
	);
	let sheenRoughnessSample = sampleLinearTexture(
		sheenRoughnessTexture,
		sheenRoughnessSampler,
		TEX_SHEEN_ROUGHNESS,
		projectorUV
	);
	let transmissionSample = sampleLinearTexture(
		transmissionTexture,
		transmissionSampler,
		TEX_TRANSMISSION,
		projectorUV
	);
	let thicknessSample = sampleLinearTexture(
		thicknessTexture,
		transmissionSampler,
		TEX_THICKNESS,
		projectorUV
	);
	let iridescenceSample = textureSample(
		iridescenceTexture,
		transmissionSampler,
		transformUV(TEX_IRIDESCENCE, projectorUV)
	);
	let iridescenceThicknessSample = textureSample(
		iridescenceThicknessTexture,
		transmissionSampler,
		transformUV(TEX_IRIDESCENCE_THICKNESS, projectorUV)
	);

	let decalNormalLocal = safeNormalize(
		vec3<f32>(
			(normalSample.rg * 2.0 - vec2<f32>(1.0)) * decal.surfaceParams1.y,
			normalSample.b * 2.0 - 1.0
		),
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let decalNormal = transformLocalDirection(decalNormalLocal);
	let clearcoatNormalLocal = safeNormalize(
		vec3<f32>(
			(clearcoatNormalSample.rg * 2.0 - vec2<f32>(1.0)) *
				decal.sheenColorClearcoatNormalScale.a,
			clearcoatNormalSample.b * 2.0 - 1.0
		),
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let clearcoatNormal = transformLocalDirection(clearcoatNormalLocal);

	let roughness = clamp(decal.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let metalness = clamp(decal.surfaceParams0.y * mrSample.b, 0.0, 1.0);
	let emissive = max(
		decal.emissiveFactor.rgb * emissiveSample.rgb * decal.emissiveFactor.a,
		vec3<f32>(0.0)
	);
	let occlusion = clamp(1.0 + decal.surfaceParams1.x * (occlusionSample.r - 1.0), 0.0, 1.0);
	let specularFactor = clamp(decal.specularColorFactor.a * specularSample.a, 0.0, 1.0);
	let specularColor = clamp(
		decal.specularColorFactor.rgb * specularColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);
	let clearcoat = clamp(decal.surfaceParams1.z * clearcoatSample.r, 0.0, 1.0);
	let clearcoatRoughness = clamp(
		decal.surfaceParams1.w * clearcoatRoughnessSample.g,
		0.04,
		1.0
	);
	let sheenColor = clamp(
		decal.sheenColorClearcoatNormalScale.rgb * sheenColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);
	let sheenRoughness = clamp(decal.surfaceParams2.x * sheenRoughnessSample.a, 0.0, 1.0);
	let transmission = clamp(decal.surfaceParams2.y * transmissionSample.r, 0.0, 1.0);
	let thickness = max(decal.surfaceParams2.w * thicknessSample.g, 0.0);
	let iridescence = clamp(decal.surfaceParams3.y * iridescenceSample.r, 0.0, 1.0);
	let iridescenceThickness = max(
		mix(
			decal.surfaceParams3.w,
			decal.attenuationColor.a,
			iridescenceThicknessSample.g
		),
		0.0
	);

	var anisotropyDirection = vec2<f32>(1.0, 0.0);
	var anisotropyStrength = clamp(decal.anisotropyParams.x, 0.0, 1.0);
	if (decal.anisotropyTextureTransformB.w > 0.5) {
		let anisotropySample = textureSample(
			anisotropyTexture,
			transmissionSampler,
			transformAnisotropyUV(projectorUV)
		);
		anisotropyDirection = anisotropySample.rg * 2.0 - vec2<f32>(1.0);
		anisotropyStrength = clamp(anisotropyStrength * anisotropySample.b, 0.0, 1.0);
	}
	let anisotropyLocal = safeNormalize(
		vec3<f32>(anisotropyDirection.x, anisotropyDirection.y, 0.0),
		vec3<f32>(1.0, 0.0, 0.0)
	);
	let anisotropyTangent = transformLocalDirection(anisotropyLocal);

	var albedoAlpha = albedoAlphaOld;
	var normalRoughMetal = normalRoughMetalOld;
	var emissiveOcclusion = emissiveOcclusionOld;
	let motionDepth = motionDepthOld;
	var specular = specularOld;
	var coatSheen = coatSheenOld;
	var sheenReflectance = sheenReflectanceOld;
	var materialExt0 = materialExt0Old;
	let materialExt1 = materialExt1Old;
	var materialExt2 = materialExt2Old;
	var materialExt3 = materialExt3Old;

	albedoAlpha = vec4<f32>(
		clamp(blendVec3(albedoAlpha.rgb, baseColor, getChannelMode(0u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		albedoAlpha.a
	);
	let blendedNormal = blendNormal(oldNormal, decalNormal, getChannelMode(1u), opacity);
	normalRoughMetal = vec4<f32>(
		encodeNormalForGBuffer(blendedNormal),
		clamp(blendScalar(normalRoughMetal.z, roughness, getChannelMode(2u), opacity), 0.04, 1.0),
		clamp(blendScalar(normalRoughMetal.w, metalness, getChannelMode(3u), opacity), 0.0, 1.0)
	);
	emissiveOcclusion = vec4<f32>(
		max(blendVec3(emissiveOcclusion.rgb, emissive, getChannelMode(4u), opacity), vec3<f32>(0.0)),
		clamp(blendScalar(emissiveOcclusion.a, occlusion, getChannelMode(5u), opacity), 0.0, 1.0)
	);
	specular = vec4<f32>(
		clamp(blendVec3(specular.rgb, specularColor, getChannelMode(7u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(blendScalar(specular.a, specularFactor, getChannelMode(6u), opacity), 0.0, 1.0)
	);
	coatSheen = vec4<f32>(
		clamp(blendScalar(coatSheen.x, clearcoat, getChannelMode(8u), opacity), 0.0, 1.0),
		clamp(blendScalar(coatSheen.y, clearcoatRoughness, getChannelMode(9u), opacity), 0.04, 1.0),
		clamp(blendScalar(coatSheen.z, sheenRoughness, getChannelMode(12u), opacity), 0.0, 1.0),
		clamp(blendScalar(coatSheen.w, transmission, getChannelMode(13u), opacity), 0.0, 1.0)
	);
	sheenReflectance = vec4<f32>(
		clamp(blendVec3(sheenReflectance.rgb, sheenColor, getChannelMode(11u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		sheenReflectance.a
	);
	let blendedClearcoatNormal = blendNormal(
		decodeDeferredNormal(materialExt0.xy),
		clearcoatNormal,
		getChannelMode(10u),
		opacity
	);
	materialExt0 = vec4<f32>(
		encodeNormalForGBuffer(blendedClearcoatNormal),
		materialExt0.z,
		max(blendScalar(materialExt0.w, thickness, getChannelMode(14u), opacity), 0.0)
	);
	materialExt2 = vec4<f32>(
		clamp(blendScalar(materialExt2.x, iridescence, getChannelMode(15u), opacity), 0.0, 1.0),
		materialExt2.y,
		max(blendScalar(materialExt2.z, iridescenceThickness, getChannelMode(16u), opacity), 0.0),
		materialExt2.w
	);
	let blendedAnisotropyTangent = blendNormal(
		decodeDeferredNormal(materialExt3.xy),
		anisotropyTangent,
		getChannelMode(17u),
		opacity
	);
	materialExt3 = vec4<f32>(
		encodeNormalForGBuffer(blendedAnisotropyTangent),
		clamp(blendScalar(materialExt3.z, anisotropyStrength, getChannelMode(17u), opacity), 0.0, 1.0),
		materialExt3.w
	);

	textureStore(gMaterialExt0Out, coord, materialExt0);
	textureStore(gMaterialExt1Out, coord, materialExt1);
	textureStore(gMaterialExt2Out, coord, materialExt2);
	textureStore(gMaterialExt3Out, coord, materialExt3);

	var output: GBufferOutput;
	output.gAlbedoAlpha = albedoAlpha;
	output.gNormalRoughMetal = normalRoughMetal;
	output.gEmissiveOcclusion = emissiveOcclusion;
	output.gMotionDepth = motionDepth;
	output.gSpecular = specular;
	output.gCoatSheen = coatSheen;
	output.gSheenReflectance = sheenReflectance;
	return output;
}

fn transformUVFrom(d: DecalUniforms, slotIndex: u32, uv: vec2<f32>) -> vec2<f32> {
	let transformA = d.textureTransformA[slotIndex];
	let transformB = d.textureTransformB[slotIndex];
	var transformed = uv * transformA.zw;
	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		transformed = vec2<f32>(
			transformed.x * c - transformed.y * s,
			transformed.x * s + transformed.y * c
		);
	}
	return transformed + transformA.xy;
}

fn transformAnisotropyUVFrom(d: DecalUniforms, uv: vec2<f32>) -> vec2<f32> {
	let transformA = d.anisotropyTextureTransformA;
	let transformB = d.anisotropyTextureTransformB;
	var transformed = uv * transformA.zw;
	let rotation = transformB.x;
	if (abs(rotation) > EPSILON) {
		let c = cos(rotation);
		let s = sin(rotation);
		transformed = vec2<f32>(
			transformed.x * c - transformed.y * s,
			transformed.x * s + transformed.y * c
		);
	}
	return transformed + transformA.xy;
}

fn sampleLinearTextureFrom(
	d: DecalUniforms,
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv: vec2<f32>
) -> vec4<f32> {
	return textureSample(textureRef, samplerRef, transformUVFrom(d, slotIndex, uv));
}

fn sampleColorTextureFrom(
	d: DecalUniforms,
	textureRef: texture_2d<f32>,
	samplerRef: sampler,
	slotIndex: u32,
	uv: vec2<f32>
) -> vec4<f32> {
	let sampled = sampleLinearTextureFrom(d, textureRef, samplerRef, slotIndex, uv);
	let isLinear = d.textureTransformB[slotIndex].z > 0.5;
	return select(vec4<f32>(srgbToLinear(sampled.rgb), sampled.a), sampled, isLinear);
}

fn transformLocalDirectionFrom(d: DecalUniforms, direction: vec3<f32>) -> vec3<f32> {
	return safeNormalize((d.localToWorld * vec4<f32>(direction, 0.0)).xyz, direction);
}

fn getChannelModeFrom(d: DecalUniforms, index: u32) -> u32 {
	let vectorIndex = index / 4u;
	let componentIndex = index % 4u;
	let values = d.channelModes[vectorIndex];
	let value = select(
		select(values.x, values.y, componentIndex == 1u),
		select(values.z, values.w, componentIndex == 3u),
		componentIndex >= 2u
	);
	return u32(clamp(floor(value + 0.5), 0.0, 5.0));
}

fn projectorOpacityFrom(
	d: DecalUniforms,
	localPosition: vec3<f32>,
	baseAlpha: f32
) -> f32 {
	let halfExtent = vec3<f32>(0.5);
	let distanceToEdge = min(
		min(halfExtent.x - abs(localPosition.x), halfExtent.y - abs(localPosition.y)),
		halfExtent.z - abs(localPosition.z)
	);
	if (distanceToEdge < 0.0) {
		return 0.0;
	}
	let edgeFade = max(d.projectorParams.y, 0.0);
	let fade = select(1.0, saturate(distanceToEdge / max(edgeFade, EPSILON)), edgeFade > 0.0);
	return saturate(d.projectorParams.x * baseAlpha * fade);
}

fn makeDecalEvaluation(
	applied: u32,
	gAlbedoAlpha: vec4<f32>,
	gNormalRoughMetal: vec4<f32>,
	gEmissiveOcclusion: vec4<f32>,
	gMotionDepth: vec4<f32>,
	gSpecular: vec4<f32>,
	gCoatSheen: vec4<f32>,
	gSheenReflectance: vec4<f32>,
	gMaterialExt0: vec4<f32>,
	gMaterialExt1: vec4<f32>,
	gMaterialExt2: vec4<f32>,
	gMaterialExt3: vec4<f32>
) -> DecalEvaluation {
	var result: DecalEvaluation;
	result.applied = applied;
	result.gAlbedoAlpha = gAlbedoAlpha;
	result.gNormalRoughMetal = gNormalRoughMetal;
	result.gEmissiveOcclusion = gEmissiveOcclusion;
	result.gMotionDepth = gMotionDepth;
	result.gSpecular = gSpecular;
	result.gCoatSheen = gCoatSheen;
	result.gSheenReflectance = gSheenReflectance;
	result.gMaterialExt0 = gMaterialExt0;
	result.gMaterialExt1 = gMaterialExt1;
	result.gMaterialExt2 = gMaterialExt2;
	result.gMaterialExt3 = gMaterialExt3;
	return result;
}

fn applyDecalToGBuffer(
	d: DecalUniforms,
	coord: vec2<i32>,
	albedoAlphaOld: vec4<f32>,
	normalRoughMetalOld: vec4<f32>,
	emissiveOcclusionOld: vec4<f32>,
	motionDepthOld: vec4<f32>,
	specularOld: vec4<f32>,
	coatSheenOld: vec4<f32>,
	sheenReflectanceOld: vec4<f32>,
	materialExt0Old: vec4<f32>,
	materialExt1Old: vec4<f32>,
	materialExt2Old: vec4<f32>,
	materialExt3Old: vec4<f32>
) -> DecalEvaluation {
	if (motionDepthOld.z <= 0.0 || !layerMatches(materialExt3Old.w, d.projectorParams.z)) {
		return makeDecalEvaluation(
			0u,
			albedoAlphaOld,
			normalRoughMetalOld,
			emissiveOcclusionOld,
			motionDepthOld,
			specularOld,
			coatSheenOld,
			sheenReflectanceOld,
			materialExt0Old,
			materialExt1Old,
			materialExt2Old,
			materialExt3Old
		);
	}

	let dimensions = vec2<f32>(textureDimensions(gAlbedoAlphaIn, 0));
	let screenUV = vec2<f32>(coord) / max(dimensions, vec2<f32>(1.0));
	let worldPosition = reconstructDeferredWorldPosition(screenUV, motionDepthOld.z);
	let localPosition = (d.worldToLocal * vec4<f32>(worldPosition, 1.0)).xyz;
	let projectorUV = localPosition.xy + vec2<f32>(0.5);
	let baseSample = sampleColorTextureFrom(
		d,
		baseColorTexture,
		baseColorSampler,
		TEX_BASE_COLOR,
		projectorUV
	);
	let opacity = projectorOpacityFrom(d, localPosition, baseSample.a);
	if (opacity <= 0.0) {
		return makeDecalEvaluation(
			0u,
			albedoAlphaOld,
			normalRoughMetalOld,
			emissiveOcclusionOld,
			motionDepthOld,
			specularOld,
			coatSheenOld,
			sheenReflectanceOld,
			materialExt0Old,
			materialExt1Old,
			materialExt2Old,
			materialExt3Old
		);
	}

	let oldNormal = decodeDeferredNormal(normalRoughMetalOld.xy);
	let baseColor = clamp(d.baseColorFactor.rgb * baseSample.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
	let mrSample = sampleLinearTextureFrom(
		d,
		metallicRoughnessTexture,
		metallicRoughnessSampler,
		TEX_METALLIC_ROUGHNESS,
		projectorUV
	);
	let normalSample = sampleLinearTextureFrom(
		d,
		normalTexture,
		normalSampler,
		TEX_NORMAL,
		projectorUV
	).rgb;
	let emissiveSample = sampleColorTextureFrom(
		d,
		emissiveTexture,
		emissiveSampler,
		TEX_EMISSIVE,
		projectorUV
	);
	let occlusionSample = sampleLinearTextureFrom(
		d,
		occlusionTexture,
		occlusionSampler,
		TEX_OCCLUSION,
		projectorUV
	);
	let specularSample = sampleLinearTextureFrom(
		d,
		specularTexture,
		specularSampler,
		TEX_SPECULAR,
		projectorUV
	);
	let specularColorSample = sampleColorTextureFrom(
		d,
		specularColorTexture,
		specularColorSampler,
		TEX_SPECULAR_COLOR,
		projectorUV
	);
	let clearcoatSample = sampleLinearTextureFrom(
		d,
		clearcoatTexture,
		clearcoatSampler,
		TEX_CLEARCOAT,
		projectorUV
	);
	let clearcoatRoughnessSample = sampleLinearTextureFrom(
		d,
		clearcoatRoughnessTexture,
		clearcoatRoughnessSampler,
		TEX_CLEARCOAT_ROUGHNESS,
		projectorUV
	);
	let clearcoatNormalSample = sampleLinearTextureFrom(
		d,
		clearcoatNormalTexture,
		clearcoatNormalSampler,
		TEX_CLEARCOAT_NORMAL,
		projectorUV
	).rgb;
	let sheenColorSample = sampleColorTextureFrom(
		d,
		sheenColorTexture,
		sheenColorSampler,
		TEX_SHEEN_COLOR,
		projectorUV
	);
	let sheenRoughnessSample = sampleLinearTextureFrom(
		d,
		sheenRoughnessTexture,
		sheenRoughnessSampler,
		TEX_SHEEN_ROUGHNESS,
		projectorUV
	);
	let transmissionSample = sampleLinearTextureFrom(
		d,
		transmissionTexture,
		transmissionSampler,
		TEX_TRANSMISSION,
		projectorUV
	);
	let thicknessSample = sampleLinearTextureFrom(
		d,
		thicknessTexture,
		transmissionSampler,
		TEX_THICKNESS,
		projectorUV
	);
	let iridescenceSample = textureSample(
		iridescenceTexture,
		transmissionSampler,
		transformUVFrom(d, TEX_IRIDESCENCE, projectorUV)
	);
	let iridescenceThicknessSample = textureSample(
		iridescenceThicknessTexture,
		transmissionSampler,
		transformUVFrom(d, TEX_IRIDESCENCE_THICKNESS, projectorUV)
	);

	let decalNormalLocal = safeNormalize(
		vec3<f32>(
			(normalSample.rg * 2.0 - vec2<f32>(1.0)) * d.surfaceParams1.y,
			normalSample.b * 2.0 - 1.0
		),
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let decalNormal = transformLocalDirectionFrom(d, decalNormalLocal);
	let clearcoatNormalLocal = safeNormalize(
		vec3<f32>(
			(clearcoatNormalSample.rg * 2.0 - vec2<f32>(1.0)) *
				d.sheenColorClearcoatNormalScale.a,
			clearcoatNormalSample.b * 2.0 - 1.0
		),
		vec3<f32>(0.0, 0.0, 1.0)
	);
	let clearcoatNormal = transformLocalDirectionFrom(d, clearcoatNormalLocal);

	let roughness = clamp(d.surfaceParams0.x * mrSample.g, 0.04, 1.0);
	let metalness = clamp(d.surfaceParams0.y * mrSample.b, 0.0, 1.0);
	let emissive = max(
		d.emissiveFactor.rgb * emissiveSample.rgb * d.emissiveFactor.a,
		vec3<f32>(0.0)
	);
	let occlusion = clamp(1.0 + d.surfaceParams1.x * (occlusionSample.r - 1.0), 0.0, 1.0);
	let specularFactor = clamp(d.specularColorFactor.a * specularSample.a, 0.0, 1.0);
	let specularColor = clamp(
		d.specularColorFactor.rgb * specularColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);
	let clearcoat = clamp(d.surfaceParams1.z * clearcoatSample.r, 0.0, 1.0);
	let clearcoatRoughness = clamp(
		d.surfaceParams1.w * clearcoatRoughnessSample.g,
		0.04,
		1.0
	);
	let sheenColor = clamp(
		d.sheenColorClearcoatNormalScale.rgb * sheenColorSample.rgb,
		vec3<f32>(0.0),
		vec3<f32>(1.0)
	);
	let sheenRoughness = clamp(d.surfaceParams2.x * sheenRoughnessSample.a, 0.0, 1.0);
	let transmission = clamp(d.surfaceParams2.y * transmissionSample.r, 0.0, 1.0);
	let thickness = max(d.surfaceParams2.w * thicknessSample.g, 0.0);
	let iridescence = clamp(d.surfaceParams3.y * iridescenceSample.r, 0.0, 1.0);
	let iridescenceThickness = max(
		mix(
			d.surfaceParams3.w,
			d.attenuationColor.a,
			iridescenceThicknessSample.g
		),
		0.0
	);

	var anisotropyDirection = vec2<f32>(1.0, 0.0);
	var anisotropyStrength = clamp(d.anisotropyParams.x, 0.0, 1.0);
	if (d.anisotropyTextureTransformB.w > 0.5) {
		let anisotropySample = textureSample(
			anisotropyTexture,
			transmissionSampler,
			transformAnisotropyUVFrom(d, projectorUV)
		);
		anisotropyDirection = anisotropySample.rg * 2.0 - vec2<f32>(1.0);
		anisotropyStrength = clamp(anisotropyStrength * anisotropySample.b, 0.0, 1.0);
	}
	let anisotropyLocal = safeNormalize(
		vec3<f32>(anisotropyDirection.x, anisotropyDirection.y, 0.0),
		vec3<f32>(1.0, 0.0, 0.0)
	);
	let anisotropyTangent = transformLocalDirectionFrom(d, anisotropyLocal);

	var albedoAlpha = albedoAlphaOld;
	var normalRoughMetal = normalRoughMetalOld;
	var emissiveOcclusion = emissiveOcclusionOld;
	let motionDepth = motionDepthOld;
	var specular = specularOld;
	var coatSheen = coatSheenOld;
	var sheenReflectance = sheenReflectanceOld;
	var materialExt0 = materialExt0Old;
	let materialExt1 = materialExt1Old;
	var materialExt2 = materialExt2Old;
	var materialExt3 = materialExt3Old;

	albedoAlpha = vec4<f32>(
		clamp(blendVec3(albedoAlpha.rgb, baseColor, getChannelModeFrom(d, 0u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		albedoAlpha.a
	);
	let blendedNormal = blendNormal(oldNormal, decalNormal, getChannelModeFrom(d, 1u), opacity);
	normalRoughMetal = vec4<f32>(
		encodeNormalForGBuffer(blendedNormal),
		clamp(blendScalar(normalRoughMetal.z, roughness, getChannelModeFrom(d, 2u), opacity), 0.04, 1.0),
		clamp(blendScalar(normalRoughMetal.w, metalness, getChannelModeFrom(d, 3u), opacity), 0.0, 1.0)
	);
	emissiveOcclusion = vec4<f32>(
		max(blendVec3(emissiveOcclusion.rgb, emissive, getChannelModeFrom(d, 4u), opacity), vec3<f32>(0.0)),
		clamp(blendScalar(emissiveOcclusion.a, occlusion, getChannelModeFrom(d, 5u), opacity), 0.0, 1.0)
	);
	specular = vec4<f32>(
		clamp(blendVec3(specular.rgb, specularColor, getChannelModeFrom(d, 7u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		clamp(blendScalar(specular.a, specularFactor, getChannelModeFrom(d, 6u), opacity), 0.0, 1.0)
	);
	coatSheen = vec4<f32>(
		clamp(blendScalar(coatSheen.x, clearcoat, getChannelModeFrom(d, 8u), opacity), 0.0, 1.0),
		clamp(blendScalar(coatSheen.y, clearcoatRoughness, getChannelModeFrom(d, 9u), opacity), 0.04, 1.0),
		clamp(blendScalar(coatSheen.z, sheenRoughness, getChannelModeFrom(d, 12u), opacity), 0.0, 1.0),
		clamp(blendScalar(coatSheen.w, transmission, getChannelModeFrom(d, 13u), opacity), 0.0, 1.0)
	);
	sheenReflectance = vec4<f32>(
		clamp(blendVec3(sheenReflectance.rgb, sheenColor, getChannelModeFrom(d, 11u), opacity), vec3<f32>(0.0), vec3<f32>(1.0)),
		sheenReflectance.a
	);
	let blendedClearcoatNormal = blendNormal(
		decodeDeferredNormal(materialExt0.xy),
		clearcoatNormal,
		getChannelModeFrom(d, 10u),
		opacity
	);
	materialExt0 = vec4<f32>(
		encodeNormalForGBuffer(blendedClearcoatNormal),
		materialExt0.z,
		max(blendScalar(materialExt0.w, thickness, getChannelModeFrom(d, 14u), opacity), 0.0)
	);
	materialExt2 = vec4<f32>(
		clamp(blendScalar(materialExt2.x, iridescence, getChannelModeFrom(d, 15u), opacity), 0.0, 1.0),
		materialExt2.y,
		max(blendScalar(materialExt2.z, iridescenceThickness, getChannelModeFrom(d, 16u), opacity), 0.0),
		materialExt2.w
	);
	let blendedAnisotropyTangent = blendNormal(
		decodeDeferredNormal(materialExt3.xy),
		anisotropyTangent,
		getChannelModeFrom(d, 17u),
		opacity
	);
	materialExt3 = vec4<f32>(
		encodeNormalForGBuffer(blendedAnisotropyTangent),
		clamp(blendScalar(materialExt3.z, anisotropyStrength, getChannelModeFrom(d, 17u), opacity), 0.0, 1.0),
		materialExt3.w
	);

	return makeDecalEvaluation(
		1u,
		albedoAlpha,
		normalRoughMetal,
		emissiveOcclusion,
		motionDepth,
		specular,
		coatSheen,
		sheenReflectance,
		materialExt0,
		materialExt1,
		materialExt2,
		materialExt3
	);
}

@compute @workgroup_size(8, 8, 1)
fn csMainBatch(@builtin(global_invocation_id) globalId: vec3<u32>) {
	if (globalId.x >= batch.rect.z || globalId.y >= batch.rect.w) {
		return;
	}

	let coordU = batch.rect.xy + globalId.xy;
	let coord = vec2<i32>(i32(coordU.x), i32(coordU.y));
	let tileSize = max(batch.tileInfo.x, 1u);
	let tileColumns = max(batch.tileInfo.y, 1u);
	let tileX = globalId.x / tileSize;
	let tileY = globalId.y / tileSize;
	let tileIndex = tileY * tileColumns + tileX;
	let tileHeader = batchTileHeaders[tileIndex];
	let decalOffset = tileHeader.x;
	let decalCount = tileHeader.y;

	var albedoAlpha = textureLoad(gAlbedoAlphaIn, coord, 0);
	var normalRoughMetal = textureLoad(gNormalRoughMetalIn, coord, 0);
	var emissiveOcclusion = textureLoad(gEmissiveOcclusionIn, coord, 0);
	var motionDepth = textureLoad(gMotionDepthIn, coord, 0);
	var specular = textureLoad(gSpecularIn, coord, 0);
	var coatSheen = textureLoad(gCoatSheenIn, coord, 0);
	var sheenReflectance = textureLoad(gSheenReflectanceIn, coord, 0);
	var materialExt0 = textureLoad(gMaterialExt0In, coord, 0);
	var materialExt1 = textureLoad(gMaterialExt1In, coord, 0);
	var materialExt2 = textureLoad(gMaterialExt2In, coord, 0);
	var materialExt3 = textureLoad(gMaterialExt3In, coord, 0);

	for (var i = 0u; i < decalCount; i = i + 1u) {
		let decalIndex = batchTileDecalIndices[decalOffset + i];
		let evaluation = applyDecalToGBuffer(
			batchDecals[decalIndex],
			coord,
			albedoAlpha,
			normalRoughMetal,
			emissiveOcclusion,
			motionDepth,
			specular,
			coatSheen,
			sheenReflectance,
			materialExt0,
			materialExt1,
			materialExt2,
			materialExt3
		);
		if (evaluation.applied != 0u) {
			albedoAlpha = evaluation.gAlbedoAlpha;
			normalRoughMetal = evaluation.gNormalRoughMetal;
			emissiveOcclusion = evaluation.gEmissiveOcclusion;
			motionDepth = evaluation.gMotionDepth;
			specular = evaluation.gSpecular;
			coatSheen = evaluation.gCoatSheen;
			sheenReflectance = evaluation.gSheenReflectance;
			materialExt0 = evaluation.gMaterialExt0;
			materialExt1 = evaluation.gMaterialExt1;
			materialExt2 = evaluation.gMaterialExt2;
			materialExt3 = evaluation.gMaterialExt3;
		}
	}

	textureStore(gAlbedoAlphaBatchOut, coord, albedoAlpha);
	textureStore(gNormalRoughMetalBatchOut, coord, normalRoughMetal);
	textureStore(gEmissiveOcclusionBatchOut, coord, emissiveOcclusion);
	textureStore(gMotionDepthBatchOut, coord, motionDepth);
	textureStore(gSpecularBatchOut, coord, specular);
	textureStore(gCoatSheenBatchOut, coord, coatSheen);
	textureStore(gSheenReflectanceBatchOut, coord, sheenReflectance);
	textureStore(gMaterialExt0Out, coord, materialExt0);
	textureStore(gMaterialExt1Out, coord, materialExt1);
	textureStore(gMaterialExt2Out, coord, materialExt2);
	textureStore(gMaterialExt3Out, coord, materialExt3);
}
