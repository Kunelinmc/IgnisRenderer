#import <ignis/webgpu/constants>

struct ParticleSimParams {
	deltaTime: f32,
	elapsedTime: f32,
	spawnCount: u32,
	maxParticles: u32,
	spawnStart: u32,
	randomSeed: u32,
	spaceMode: u32,
	receiveShadow: u32,
	gravity: vec4<f32>,
	worldPosition: vec4<f32>,
	spawnBasePosition: vec4<f32>,
	directionSpread: vec4<f32>,
	lifetimeSpeed: vec4<f32>,
	sizeRotation: vec4<f32>,
	angularAtlas: vec4<f32>,
	atlasParams: vec4<f32>,
	startColor: vec4<f32>,
}

struct ParticleState {
	positionAge: vec4<f32>,
	velocityLifetime: vec4<f32>,
	colorSize: vec4<f32>,
	rotationAngularActive: vec4<f32>,
}

struct ParticleInstance {
	positionSize: vec4<f32>,
	color: vec4<f32>,
	uvRect: vec4<f32>,
	rotation: f32,
	receiveShadow: f32,
	padding0: f32,
	padding1: f32,
}

struct ParticleStateBuffer {
	data: array<ParticleState>,
}

struct ParticleInstanceBuffer {
	data: array<ParticleInstance>,
}

struct ParticleDrawArgs {
	vertexCount: atomic<u32>,
	instanceCount: atomic<u32>,
	firstVertex: atomic<u32>,
	firstInstance: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> particles:
	ParticleStateBuffer;
@group(0) @binding(1) var<storage, read_write> instances:
	ParticleInstanceBuffer;
@group(0) @binding(2) var<storage, read_write> drawArgs:
	ParticleDrawArgs;
@group(0) @binding(3) var<uniform> params: ParticleSimParams;

fn nextRandom(seed: ptr<function, u32>) -> f32 {
	var value = (*seed);
	value = value * 1664525u + 1013904223u;
	(*seed) = value;
	return f32(value) / 4294967296.0;
}

fn randomRange(seed: ptr<function, u32>, lo: f32, hi: f32) -> f32 {
	let minValue = min(lo, hi);
	let maxValue = max(lo, hi);
	return minValue + (maxValue - minValue) * nextRandom(seed);
}

fn normalizeSafe(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
	let len = length(value);
	if (len <= EPSILON) {
		return fallback;
	}
	return value / len;
}

fn randomUnitVector(seed: ptr<function, u32>) -> vec3<f32> {
	let u = nextRandom(seed) * 2.0 - 1.0;
	let theta = nextRandom(seed) * TWO_PI;
	let r = sqrt(max(0.0, 1.0 - u * u));
	return vec3<f32>(r * cos(theta), u, r * sin(theta));
}

fn randomDirectionInCone(
	seed: ptr<function, u32>,
	direction: vec3<f32>,
	spread: f32
) -> vec3<f32> {
	let base = normalizeSafe(direction, vec3<f32>(0.0, 1.0, 0.0));
	if (spread <= 0.0) {
		return base;
	}

	let random = randomUnitVector(seed);
	let spreadScale = tan(max(0.0, spread));
	return normalizeSafe(base + random * spreadScale, base);
}

fn randomSpawnOffset(seed: ptr<function, u32>, radius: f32) -> vec3<f32> {
	if (radius <= 0.0) {
		return vec3<f32>(0.0);
	}

	let direction = randomUnitVector(seed);
	let distance = pow(max(nextRandom(seed), 0.0), 0.3333333333) * radius;
	return direction * distance;
}

fn resolveAtlasUV(age: f32) -> vec4<f32> {
	let rows = u32(max(params.angularAtlas.z, 1.0));
	let columns = u32(max(params.angularAtlas.w, 1.0));
	let fps = max(params.atlasParams.x, 0.0);
	let frameCount = max(1u, rows * columns);
	if (frameCount <= 1u || fps <= 0.0) {
		return vec4<f32>(0.0, 0.0, 1.0, 1.0);
	}

	let rawFrame = u32(max(0.0, floor(age * fps)));
	var frame = rawFrame % frameCount;
	if (params.atlasParams.y < 0.5) {
		frame = min(frameCount - 1u, rawFrame);
	}

	let column = frame % columns;
	let row = frame / columns;
	let invColumns = 1.0 / f32(columns);
	let invRows = 1.0 / f32(rows);
	let u0 = f32(column) * invColumns;
	let v0 = f32(row) * invRows;
	return vec4<f32>(u0, v0, u0 + invColumns, v0 + invRows);
}

@compute @workgroup_size(1)
fn resetMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x > 0u) {
		return;
	}

	atomicStore(&drawArgs.vertexCount, 6u);
	atomicStore(&drawArgs.instanceCount, 0u);
	atomicStore(&drawArgs.firstVertex, 0u);
	atomicStore(&drawArgs.firstInstance, 0u);
}

@compute @workgroup_size(64)
fn spawnMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let spawnIndex = gid.x;
	if (spawnIndex >= params.spawnCount || spawnIndex >= params.maxParticles) {
		return;
	}

	let slot = (params.spawnStart + spawnIndex) % params.maxParticles;
	var seed =
		params.randomSeed ^
		(spawnIndex * 747796405u) ^
		(slot * 2891336453u);
	let direction = randomDirectionInCone(
		&seed,
		params.directionSpread.xyz,
		params.directionSpread.w
	);
	let speed = randomRange(&seed, params.lifetimeSpeed.z, params.lifetimeSpeed.w);
	let velocity = direction * speed;
	let lifetime = max(
		0.01,
		randomRange(&seed, params.lifetimeSpeed.x, params.lifetimeSpeed.y)
	);
	let size = max(0.001, randomRange(&seed, params.sizeRotation.x, params.sizeRotation.y));
	let rotation = randomRange(&seed, params.sizeRotation.z, params.sizeRotation.w);
	let angularVelocity = randomRange(&seed, params.angularAtlas.x, params.angularAtlas.y);
	let spawnOffset = randomSpawnOffset(&seed, params.spawnBasePosition.w);
	let position = params.spawnBasePosition.xyz + spawnOffset;

	particles.data[slot] = ParticleState(
		vec4<f32>(position, 0.0),
		vec4<f32>(velocity, lifetime),
		vec4<f32>(params.startColor.rgb, params.startColor.a),
		vec4<f32>(rotation, angularVelocity, size, 1.0)
	);
}

@compute @workgroup_size(64)
fn simulateMain(@builtin(global_invocation_id) gid: vec3<u32>) {
	let index = gid.x;
	if (index >= params.maxParticles) {
		return;
	}

	var particle = particles.data[index];
	if (particle.rotationAngularActive.w < 0.5) {
		return;
	}

	particle.positionAge.w = particle.positionAge.w + params.deltaTime;
	if (particle.positionAge.w >= particle.velocityLifetime.w) {
		particle.rotationAngularActive.w = 0.0;
		particles.data[index] = particle;
		return;
	}

	particle.velocityLifetime.xyz =
		particle.velocityLifetime.xyz + params.gravity.xyz * params.deltaTime;
	particle.positionAge.xyz =
		particle.positionAge.xyz + particle.velocityLifetime.xyz * params.deltaTime;
	particle.rotationAngularActive.x =
		particle.rotationAngularActive.x +
		particle.rotationAngularActive.y * params.deltaTime;
	particles.data[index] = particle;

	let outputIndex = atomicAdd(&drawArgs.instanceCount, 1u);
	if (outputIndex >= params.maxParticles) {
		return;
	}

	var worldPosition = particle.positionAge.xyz;
	if (params.spaceMode == 0u) {
		worldPosition = worldPosition + params.worldPosition.xyz;
	}

	instances.data[outputIndex] = ParticleInstance(
		vec4<f32>(worldPosition, max(0.001, particle.rotationAngularActive.z)),
		particle.colorSize,
		resolveAtlasUV(particle.positionAge.w),
		particle.rotationAngularActive.x,
		f32(params.receiveShadow),
		0.0,
		0.0
	);
}
