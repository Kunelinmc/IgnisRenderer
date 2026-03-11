import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { ParticleSimulationStage } from "../src/pipeline/ParticleSimulationStage.ts";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../src/pipeline/types.ts";
import { ParticleSystem } from "../src/particles/ParticleSystem.ts";
import { ParticleSpaceMode } from "../src/particles/types.ts";

function createContext(systems) {
	const camera = new Camera();
	camera.position.set(0, 0, 10);
	camera.updateMatrices();

	return {
		camera,
		attachments: {
			width: 1280,
			height: 720,
		},
		features: {
			enableLighting: true,
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			warnings: [],
		},
		shadowMaps: new Map(),
		scene: {
			sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 10 },
			lights: [],
			particleSystems: systems,
			camera,
			skybox: null,
			models: [],
			shadowMaps: new Map(),
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			shadowTransmitterPackets: [],
			reflectivePackets: [],
		},
		shCoeffs: SH.empty(),
		shAmbientCoeffs: SH.empty(),
		worldMatrix: Matrix4.identity(),
		transient: new Map(),
	};
}

function getBatches(context) {
	return context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
}

function testDeterministicSeed() {
	const run = (seed) => {
		const stage = new ParticleSimulationStage();
		const system = new ParticleSystem({
			seed,
			space: ParticleSpaceMode.World,
			maxParticles: 16,
			emit: {
				rate: 0,
				bursts: [{ time: 0, count: 4 }],
				lifetimeRange: [3, 3],
				speedRange: [2, 2],
				sizeRange: [1, 1],
				spawnRadius: 1,
				spread: 1.2,
			},
		});
		const context = createContext([system]);
		stage.execute(context, 0.016);
		const particles = getBatches(context)[0]?.particles ?? [];
		return particles.map((particle) => ({
			x: Number(particle.position.x.toFixed(6)),
			y: Number(particle.position.y.toFixed(6)),
			z: Number(particle.position.z.toFixed(6)),
		}));
	};

	assert.deepEqual(run(42), run(42));
	assert.notDeepEqual(run(42), run(1337));
}

function testRateAndBurstSpawn() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		maxParticles: 64,
		emit: {
			rate: 2,
			bursts: [{ time: 0, count: 3 }],
			lifetimeRange: [10, 10],
			speedRange: [0, 0],
		},
	});
	const context = createContext([system]);
	stage.execute(context, 1);
	const particles = getBatches(context)[0]?.particles ?? [];
	assert.equal(particles.length, 5);
}

function testGradientAndAtlas() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		maxParticles: 8,
		atlas: {
			rows: 2,
			columns: 2,
			fps: 1,
			loop: false,
		},
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 1 }],
			lifetimeRange: [2, 2],
			speedRange: [0, 0],
			sizeRange: [4, 4],
		},
		sizeOverLifetime: [
			{ t: 0, value: 1 },
			{ t: 1, value: 3 },
		],
		colorOverLifetime: [
			{ t: 0, value: { r: 255, g: 0, b: 0, a: 1 } },
			{ t: 1, value: { r: 0, g: 0, b: 255, a: 0.5 } },
		],
	});

	const context = createContext([system]);
	stage.execute(context, 1);
	const particle = getBatches(context)[0]?.particles?.[0];
	assert.ok(particle);
	assert.ok(Math.abs(particle.size - 8) < 1e-6);
	assert.ok(Math.abs(particle.color.r - 127.5) < 1e-6);
	assert.ok(Math.abs(particle.color.b - 127.5) < 1e-6);
	assert.ok(Math.abs(particle.color.a - 0.75) < 1e-6);
	assert.equal(particle.uvRect.u0, 0.5);
	assert.equal(particle.uvRect.v0, 0);
}

function testLocalSpaceFollowsSystemPosition() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		space: ParticleSpaceMode.Local,
		position: { x: 4, y: 0, z: 0 },
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 1 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
		},
	});
	const context = createContext([system]);
	stage.execute(context, 0.016);
	let particle = getBatches(context)[0]?.particles?.[0];
	assert.ok(Math.abs(particle.position.x - 4) < 1e-6);

	system.position.x = 9;
	stage.execute(context, 0);
	particle = getBatches(context)[0]?.particles?.[0];
	assert.ok(Math.abs(particle.position.x - 9) < 1e-6);
}

function testCollisionAndSubEmitter() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		space: ParticleSpaceMode.World,
		position: { x: 0, y: 0.2, z: 0 },
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 1 }],
			lifetimeRange: [0.1, 0.1],
			speedRange: [2, 2],
			direction: { x: 0, y: -1, z: 0 },
			spread: 0,
		},
		colliders: [
			{
				type: "plane",
				normal: { x: 0, y: 1, z: 0 },
				constant: 0,
				restitution: 0.8,
				damping: 0.1,
			},
		],
		subEmitter: {
			enabled: true,
			trigger: "death",
			count: 2,
			inheritVelocityScale: 0,
			lifetimeRange: [2, 2],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
	});

	const context = createContext([system]);
	stage.execute(context, 0.05);
	let particles = getBatches(context)[0]?.particles ?? [];
	assert.equal(particles.length, 1);
	assert.ok(particles[0].position.y >= 0);

	stage.execute(context, 0.1);
	particles = getBatches(context)[0]?.particles ?? [];
	assert.equal(particles.length, 2);
	for (const particle of particles) {
		assert.ok(particle.position.y >= 0);
	}
}

function testLODScalesSimulationAndRenderSubset() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		maxParticles: 10,
		position: { x: 0, y: 0, z: -200 },
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 8 }],
			lifetimeRange: [5, 5],
			speedRange: [0, 0],
			sizeRange: [1, 1],
			spawnRadius: 0,
		},
		lod: {
			enabled: true,
			hysteresisFrames: 0,
			levels: [
				{
					distance: 16,
					projectedSize: 128,
					simulationIntervalFrames: 1,
					spawnScale: 1,
					maxParticlesScale: 1,
					renderSortRatio: 1,
				},
				{
					distance: Number.POSITIVE_INFINITY,
					projectedSize: 0,
					simulationIntervalFrames: 2,
					spawnScale: 0.5,
					maxParticlesScale: 0.5,
					renderSortRatio: 0.5,
				},
			],
		},
	});

	const context = createContext([system]);
	stage.execute(context, 0.016);
	let particles = getBatches(context)[0]?.particles ?? [];
	assert.equal(particles.length, 0);

	stage.execute(context, 0.016);
	particles = getBatches(context)[0]?.particles ?? [];
	assert.equal(particles.length, 2);
}

function testLODHysteresis() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		maxParticles: 128,
		position: { x: 0, y: 0, z: 8 },
		emit: {
			rate: 10,
			bursts: [],
			lifetimeRange: [10, 10],
			speedRange: [0, 0],
			sizeRange: [1, 1],
		},
		lod: {
			enabled: true,
			hysteresisFrames: 2,
			levels: [
				{
					distance: 16,
					projectedSize: 16,
					simulationIntervalFrames: 1,
					spawnScale: 1,
					maxParticlesScale: 1,
					renderSortRatio: 1,
				},
				{
					distance: Number.POSITIVE_INFINITY,
					projectedSize: 0,
					simulationIntervalFrames: 1,
					spawnScale: 0,
					maxParticlesScale: 1,
					renderSortRatio: 1,
				},
			],
		},
	});

	const context = createContext([system]);
	stage.execute(context, 1);
	let total = getBatches(context)[0]?.particles?.length ?? 0;
	assert.equal(total, 10);

	system.position.z = -200;
	stage.execute(context, 1);
	total = getBatches(context)[0]?.particles?.length ?? 0;
	assert.equal(total, 20);

	stage.execute(context, 1);
	total = getBatches(context)[0]?.particles?.length ?? 0;
	assert.equal(total, 20);
}

function testStrictFailureWhenLODStillOverBudget() {
	const stage = new ParticleSimulationStage();
	const system = new ParticleSystem({
		maxParticles: 8,
		emit: {
			rate: 0,
			bursts: [{ time: 0, count: 1 }],
			lifetimeRange: [2, 2],
			speedRange: [0, 0],
		},
		lod: {
			enabled: true,
			hysteresisFrames: 0,
			levels: [
				{
					distance: Number.POSITIVE_INFINITY,
					projectedSize: 0,
					simulationIntervalFrames: 1,
					spawnScale: 1,
					maxParticlesScale: 2,
					renderSortRatio: 1,
				},
			],
		},
	});
	const context = createContext([system]);
	assert.throws(() => stage.execute(context, 0.016), /required=16 available=8/);
}

function run() {
	testDeterministicSeed();
	testRateAndBurstSpawn();
	testGradientAndAtlas();
	testLocalSpaceFollowsSystemPosition();
	testCollisionAndSubEmitter();
	testLODScalesSimulationAndRenderSubset();
	testLODHysteresis();
	testStrictFailureWhenLODStillOverBudget();
	console.log("Particle simulation stage tests passed");
}

run();
