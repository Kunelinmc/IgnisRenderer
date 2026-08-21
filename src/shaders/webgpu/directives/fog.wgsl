const IGNIS_FOG_MODE_LINEAR: i32 = 0;
const IGNIS_FOG_MODE_EXP: i32 = 1;
const IGNIS_FOG_MODE_EXP2: i32 = 2;

fn ignisFogLinear(depth: f32, startDepth: f32, endDepth: f32) -> f32 {
	let safeRange = max(endDepth - startDepth, 1e-4);
	return clamp((depth - startDepth) / safeRange, 0.0, 1.0);
}

fn ignisFogExp(depth: f32, density: f32) -> f32 {
	let d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-d);
}

fn ignisFogExp2(depth: f32, density: f32) -> f32 {
	let d = max(depth, 0.0) * max(density, 0.0);
	return 1.0 - exp(-(d * d));
}

fn ignisComputeFogFactor(
	mode: i32,
	depth: f32,
	startDepth: f32,
	endDepth: f32,
	density: f32,
	strength: f32
) -> f32 {
	var fog = ignisFogLinear(depth, startDepth, endDepth);
	if (mode == IGNIS_FOG_MODE_EXP) {
		fog = ignisFogExp(depth, density);
	} else if (mode == IGNIS_FOG_MODE_EXP2) {
		fog = ignisFogExp2(depth, density);
	}
	return clamp(fog * max(strength, 0.0), 0.0, 1.0);
}
