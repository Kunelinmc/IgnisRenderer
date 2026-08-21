fn linearToSrgb(color: vec3<f32>) -> vec3<f32> {
	let low = color * vec3<f32>(12.92);
	let high = vec3<f32>(1.055) * pow(color, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
	return select(high, low, color <= vec3<f32>(0.0031308));
}

fn srgbToLinear(color: vec3<f32>) -> vec3<f32> {
	let low = color / vec3<f32>(12.92);
	let high = pow((color + vec3<f32>(0.055)) / vec3<f32>(1.055), vec3<f32>(2.4));
	return select(high, low, color <= vec3<f32>(0.04045));
}
