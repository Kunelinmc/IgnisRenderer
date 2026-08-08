const DEFERRED_MATERIAL_MODEL_MASK: u32 = 0x3u;
const DEFERRED_MATERIAL_CLEARCOAT_BIT: u32 = 1u << 2u;
const DEFERRED_MATERIAL_SHEEN_BIT: u32 = 1u << 3u;
const DEFERRED_MATERIAL_IRIDESCENCE_BIT: u32 = 1u << 4u;
const DEFERRED_MATERIAL_ANISOTROPY_BIT: u32 = 1u << 5u;
const DEFERRED_MATERIAL_SPECULAR_BIT: u32 = 1u << 6u;
const DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT: u32 = 1u << 7u;
const DEFERRED_RENDER_LAYER_MASK: u32 = 0x7ffu;

fn encodeDeferredMaterialWord(
	shadingModel: u32,
	clearcoat: f32,
	sheenColor: vec3<f32>,
	iridescence: f32,
	anisotropyStrength: f32,
	specularColor: vec3<f32>,
	specularFactor: f32,
	reflectance: f32,
	receiveShadows: f32
) -> f32 {
	var word = shadingModel & DEFERRED_MATERIAL_MODEL_MASK;
	if (clearcoat > 1e-6) {
		word = word | DEFERRED_MATERIAL_CLEARCOAT_BIT;
	}
	if (max(max(sheenColor.x, sheenColor.y), sheenColor.z) > 1e-6) {
		word = word | DEFERRED_MATERIAL_SHEEN_BIT;
	}
	if (iridescence > 1e-6) {
		word = word | DEFERRED_MATERIAL_IRIDESCENCE_BIT;
	}
	if (anisotropyStrength > 1e-6) {
		word = word | DEFERRED_MATERIAL_ANISOTROPY_BIT;
	}
	if (
		any(abs(specularColor - vec3<f32>(1.0)) > vec3<f32>(1e-6)) ||
		abs(specularFactor - 1.0) > 1e-6 ||
		abs(reflectance - 0.5) > 1e-6
	) {
		word = word | DEFERRED_MATERIAL_SPECULAR_BIT;
	}
	if (receiveShadows > 0.5) {
		word = word | DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT;
	}
	return f32(word);
}

fn decodeDeferredMaterialWord(value: f32) -> u32 {
	return u32(max(0.0, floor(value + 0.5)));
}

fn deferredShadingModel(word: u32) -> u32 {
	return word & DEFERRED_MATERIAL_MODEL_MASK;
}

fn deferredHasFeature(word: u32, feature: u32) -> bool {
	return (word & feature) != 0u;
}

fn packDeferredExt3(
	anisotropyTangent: vec3<f32>,
	anisotropyStrength: f32,
	renderLayers: f32
) -> vec4<u32> {
	let tangent = encodeNormalForGBuffer(anisotropyTangent);
	return vec4<u32>(
		u32(round(clamp(tangent.x, 0.0, 1.0) * 65535.0)),
		u32(round(clamp(tangent.y, 0.0, 1.0) * 65535.0)),
		u32(round(clamp(anisotropyStrength, 0.0, 1.0) * 65535.0)),
		u32(max(0.0, floor(renderLayers + 0.5))) & DEFERRED_RENDER_LAYER_MASK
	);
}

fn decodeDeferredExt3Normal(value: vec2<u32>) -> vec2<f32> {
	return vec2<f32>(value) / 65535.0;
}

fn decodeDeferredExt3Strength(value: u32) -> f32 {
	return f32(value) / 65535.0;
}

fn decodeDeferredExt3RenderLayers(value: u32) -> u32 {
	return value & DEFERRED_RENDER_LAYER_MASK;
}
