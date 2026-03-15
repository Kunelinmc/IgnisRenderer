import type { IVector3 } from "../../maths/types";
import { clamp } from "../../maths/Common";
import {
	ParticleSpaceMode,
	type ParticleCollider,
	type ParticleSubEmitterConfig,
	type ParticleSystem,
} from "../../particles";
import type { RGBA } from "../../foundation/Color";
import { DEFAULT_DAMPING, DEFAULT_RESTITUTION } from "./constants";
import type { RuntimeParticle, SystemRuntimeState } from "./types";

interface SpawnOverrides {
	position?: IVector3;
	baseVelocity?: IVector3;
	lifetimeRange?: [number, number];
	speedRange?: [number, number];
	sizeRange?: [number, number];
}

export interface ParticleSimulationStepOptions {
	maxParticles: number;
	spawnScale: number;
}

export class ParticleSimulationCore {
	public createRuntime(system: ParticleSystem): SystemRuntimeState {
		return {
			particles: [],
			emissionRemainder: 0,
			elapsed: 0,
			burstCycles: [],
			randomState: system.seed >>> 0 || 1,
			frameIndex: 0,
			pendingSimulationTime: 0,
			lodLevelIndex: 0,
			lodCandidateLevelIndex: -1,
			lodCandidateFrameCount: 0,
			activeLODLevel: null,
		};
	}

	public trimToMaxParticles(
		runtime: SystemRuntimeState,
		maxParticles: number
	): void {
		const cap = Math.max(1, Math.floor(maxParticles));
		const overflow = runtime.particles.length - cap;
		if (overflow > 0) {
			runtime.particles.splice(0, overflow);
		}
	}

	public stepSystem(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		dt: number,
		options: ParticleSimulationStepOptions
	): void {
		if (dt <= 0) return;

		const spawnScale = Math.max(0, options.spawnScale);
		const maxParticles = Math.max(1, Math.floor(options.maxParticles));
		runtime.elapsed += dt;

		const spawnCount = this._resolveSpawnCount(system, runtime, dt, spawnScale);
		if (spawnCount > 0) {
			this._spawnParticles(system, runtime, spawnCount, maxParticles);
		}

		const colliders = system.colliders ?? [];
		const gravity = system.gravity;
		const subEmitter = system.subEmitter;

		for (let i = runtime.particles.length - 1; i >= 0; i--) {
			const particle = runtime.particles[i];
			particle.age += dt;
			if (particle.age >= particle.lifetime) {
				if (subEmitter?.enabled !== false && subEmitter) {
					this._spawnSubEmitterParticles(
						system,
						runtime,
						particle,
						subEmitter,
						spawnScale,
						maxParticles
					);
				}
				runtime.particles.splice(i, 1);
				continue;
			}

			particle.velocity.x += gravity.x * dt;
			particle.velocity.y += gravity.y * dt;
			particle.velocity.z += gravity.z * dt;

			particle.position.x += particle.velocity.x * dt;
			particle.position.y += particle.velocity.y * dt;
			particle.position.z += particle.velocity.z * dt;
			particle.rotation += particle.angularVelocity * dt;

			if (colliders.length === 0) continue;
			for (const collider of colliders) {
				if (collider.enabled === false) continue;
				this._resolveCollider(particle, collider);
			}
		}
	}

	private _resolveSpawnCount(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		dt: number,
		spawnScale: number
	): number {
		const emit = system.emit ?? {};
		const rate = Math.max(0, emit.rate ?? 0) * spawnScale;
		runtime.emissionRemainder += rate * dt;

		let spawnCount = Math.floor(runtime.emissionRemainder);
		if (spawnCount > 0) {
			runtime.emissionRemainder -= spawnCount;
		}

		const bursts = emit.bursts ?? [];
		for (let i = 0; i < bursts.length; i++) {
			const burst = bursts[i];
			const cycles = Math.max(1, burst.cycles ?? 1);
			const interval = Math.max(0, burst.interval ?? 0);
			let firedCycles = runtime.burstCycles[i] ?? 0;

			while (firedCycles < cycles) {
				const triggerAt = burst.time + interval * firedCycles;
				if (runtime.elapsed < triggerAt) break;
				const burstCount = Math.max(
					0,
					Math.floor(Math.max(0, burst.count) * spawnScale)
				);
				spawnCount += burstCount;
				firedCycles++;
			}

			runtime.burstCycles[i] = firedCycles;
		}

		return spawnCount;
	}

	private _spawnParticles(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		count: number,
		maxParticles: number,
		overrides: SpawnOverrides = {}
	): void {
		const available = Math.max(0, maxParticles - runtime.particles.length);
		const spawnCount = Math.min(available, Math.max(0, count));
		if (spawnCount <= 0) return;

		for (let i = 0; i < spawnCount; i++) {
			runtime.particles.push(this._createParticle(system, runtime, overrides));
		}
	}

	private _spawnSubEmitterParticles(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		parent: RuntimeParticle,
		subEmitter: ParticleSubEmitterConfig,
		spawnScale: number,
		maxParticles: number
	): void {
		if ((subEmitter.trigger ?? "death") !== "death") return;

		const count = Math.max(
			0,
			Math.floor(Math.max(0, subEmitter.count ?? 0) * spawnScale)
		);
		if (count <= 0) return;

		const inheritScale = subEmitter.inheritVelocityScale ?? 0.5;
		const baseVelocity = {
			x: parent.velocity.x * inheritScale,
			y: parent.velocity.y * inheritScale,
			z: parent.velocity.z * inheritScale,
		};
		this._spawnParticles(system, runtime, count, maxParticles, {
			position: parent.position,
			baseVelocity,
			lifetimeRange: subEmitter.lifetimeRange,
			speedRange: subEmitter.speedRange,
			sizeRange: subEmitter.sizeRange,
		});
	}

	private _createParticle(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		overrides: SpawnOverrides = {}
	): RuntimeParticle {
		const emit = system.emit;
		const lifetimeRange = overrides.lifetimeRange ??
			emit.lifetimeRange ?? [0.5, 1.5];
		const speedRange = overrides.speedRange ?? emit.speedRange ?? [2, 5];
		const sizeRange = overrides.sizeRange ?? emit.sizeRange ?? [0.5, 1.0];
		const rotationRange = emit.rotationRange ?? [0, 0];
		const angularVelocityRange = emit.angularVelocityRange ?? [0, 0];
		const startColor = emit.startColor ?? { r: 255, g: 255, b: 255, a: 1 };

		const randomDirection = this._randomDirectionInCone(
			runtime,
			emit.direction ?? { x: 0, y: 1, z: 0 },
			Math.max(0, emit.spread ?? 0)
		);
		const speed = this._randomRange(runtime, speedRange[0], speedRange[1]);
		const spawnedVelocity = {
			x: randomDirection.x * speed,
			y: randomDirection.y * speed,
			z: randomDirection.z * speed,
		};
		const velocity =
			overrides.baseVelocity ?
				{
					x: spawnedVelocity.x + overrides.baseVelocity.x,
					y: spawnedVelocity.y + overrides.baseVelocity.y,
					z: spawnedVelocity.z + overrides.baseVelocity.z,
				}
			:	spawnedVelocity;

		const spawnOffset =
			overrides.position ?
				{ x: 0, y: 0, z: 0 }
			:	this._randomSpawnOffset(runtime, Math.max(0, emit.spawnRadius ?? 0));
		const spawnBasePosition =
			overrides.position ? overrides.position
			: system.space === ParticleSpaceMode.Local ? { x: 0, y: 0, z: 0 }
			: system.getWorldPosition();

		return {
			position: {
				x: spawnBasePosition.x + spawnOffset.x,
				y: spawnBasePosition.y + spawnOffset.y,
				z: spawnBasePosition.z + spawnOffset.z,
			},
			velocity,
			age: 0,
			lifetime: Math.max(
				0.01,
				this._randomRange(runtime, lifetimeRange[0], lifetimeRange[1])
			),
			startSize: Math.max(
				0.001,
				this._randomRange(runtime, sizeRange[0], sizeRange[1])
			),
			startColor: {
				r: startColor.r,
				g: startColor.g,
				b: startColor.b,
				a: startColor.a,
			},
			rotation: this._randomRange(runtime, rotationRange[0], rotationRange[1]),
			angularVelocity: this._randomRange(
				runtime,
				angularVelocityRange[0],
				angularVelocityRange[1]
			),
		};
	}

	private _resolveCollider(
		particle: RuntimeParticle,
		collider: ParticleCollider
	): void {
		switch (collider.type) {
			case "plane":
				this._resolvePlaneCollider(particle, collider);
				break;
			case "sphere":
				this._resolveSphereCollider(particle, collider);
				break;
			case "aabb":
				this._resolveAABBCollider(particle, collider);
				break;
		}
	}

	private _resolvePlaneCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "plane" }>
	): void {
		const normal = normalizeVector(collider.normal);
		const distance =
			particle.position.x * normal.x +
			particle.position.y * normal.y +
			particle.position.z * normal.z +
			collider.constant;
		if (distance >= 0) return;

		particle.position.x -= normal.x * distance;
		particle.position.y -= normal.y * distance;
		particle.position.z -= normal.z * distance;
		this._applyBounce(particle, normal, collider.restitution, collider.damping);
	}

	private _resolveSphereCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "sphere" }>
	): void {
		const dx = particle.position.x - collider.center.x;
		const dy = particle.position.y - collider.center.y;
		const dz = particle.position.z - collider.center.z;
		const radius = Math.max(0.001, collider.radius);
		const distanceSq = dx * dx + dy * dy + dz * dz;
		if (distanceSq >= radius * radius) return;

		const distance = Math.sqrt(distanceSq) || 1;
		const normal = {
			x: dx / distance,
			y: dy / distance,
			z: dz / distance,
		};

		particle.position.x = collider.center.x + normal.x * radius;
		particle.position.y = collider.center.y + normal.y * radius;
		particle.position.z = collider.center.z + normal.z * radius;
		this._applyBounce(particle, normal, collider.restitution, collider.damping);
	}

	private _resolveAABBCollider(
		particle: RuntimeParticle,
		collider: Extract<ParticleCollider, { type: "aabb" }>
	): void {
		const p = particle.position;
		if (
			p.x < collider.min.x ||
			p.x > collider.max.x ||
			p.y < collider.min.y ||
			p.y > collider.max.y ||
			p.z < collider.min.z ||
			p.z > collider.max.z
		) {
			return;
		}

		const distances = [
			{
				axis: "x",
				value: Math.abs(p.x - collider.min.x),
				normal: { x: -1, y: 0, z: 0 },
			},
			{
				axis: "x",
				value: Math.abs(collider.max.x - p.x),
				normal: { x: 1, y: 0, z: 0 },
			},
			{
				axis: "y",
				value: Math.abs(p.y - collider.min.y),
				normal: { x: 0, y: -1, z: 0 },
			},
			{
				axis: "y",
				value: Math.abs(collider.max.y - p.y),
				normal: { x: 0, y: 1, z: 0 },
			},
			{
				axis: "z",
				value: Math.abs(p.z - collider.min.z),
				normal: { x: 0, y: 0, z: -1 },
			},
			{
				axis: "z",
				value: Math.abs(collider.max.z - p.z),
				normal: { x: 0, y: 0, z: 1 },
			},
		];
		distances.sort((left, right) => left.value - right.value);

		const hit = distances[0];
		if (!hit) return;

		if (hit.axis === "x") {
			p.x = hit.normal.x < 0 ? collider.min.x : collider.max.x;
		} else if (hit.axis === "y") {
			p.y = hit.normal.y < 0 ? collider.min.y : collider.max.y;
		} else {
			p.z = hit.normal.z < 0 ? collider.min.z : collider.max.z;
		}

		this._applyBounce(
			particle,
			hit.normal,
			collider.restitution,
			collider.damping
		);
	}

	private _applyBounce(
		particle: RuntimeParticle,
		normal: IVector3,
		restitution: number | undefined,
		damping: number | undefined
	): void {
		const rest = clamp(restitution ?? DEFAULT_RESTITUTION);
		const damp = clamp(damping ?? DEFAULT_DAMPING);
		const velocity = particle.velocity;
		const vn =
			velocity.x * normal.x + velocity.y * normal.y + velocity.z * normal.z;
		if (vn < 0) {
			const impulse = -(1 + rest) * vn;
			velocity.x += normal.x * impulse;
			velocity.y += normal.y * impulse;
			velocity.z += normal.z * impulse;
		}
		const dampingScale = 1 - damp;
		velocity.x *= dampingScale;
		velocity.y *= dampingScale;
		velocity.z *= dampingScale;
	}

	private _randomDirectionInCone(
		runtime: SystemRuntimeState,
		direction: IVector3,
		spread: number
	): IVector3 {
		const base = normalizeVector(direction);
		if (spread <= 0) return base;

		const random = this._randomUnitVector(runtime);
		const spreadScale = Math.tan(spread);
		return normalizeVector({
			x: base.x + random.x * spreadScale,
			y: base.y + random.y * spreadScale,
			z: base.z + random.z * spreadScale,
		});
	}

	private _randomSpawnOffset(
		runtime: SystemRuntimeState,
		radius: number
	): IVector3 {
		if (radius <= 0) return { x: 0, y: 0, z: 0 };
		const direction = this._randomUnitVector(runtime);
		const distance = Math.cbrt(this._nextRandom(runtime)) * radius;
		return {
			x: direction.x * distance,
			y: direction.y * distance,
			z: direction.z * distance,
		};
	}

	private _randomUnitVector(runtime: SystemRuntimeState): IVector3 {
		const u = this._nextRandom(runtime) * 2 - 1;
		const theta = this._nextRandom(runtime) * Math.PI * 2;
		const r = Math.sqrt(Math.max(0, 1 - u * u));
		return {
			x: r * Math.cos(theta),
			y: u,
			z: r * Math.sin(theta),
		};
	}

	private _randomRange(
		runtime: SystemRuntimeState,
		min: number,
		max: number
	): number {
		const lo = Math.min(min, max);
		const hi = Math.max(min, max);
		return lo + (hi - lo) * this._nextRandom(runtime);
	}

	private _nextRandom(runtime: SystemRuntimeState): number {
		runtime.randomState = (runtime.randomState * 1664525 + 1013904223) >>> 0;
		return runtime.randomState / 0x100000000;
	}
}

function normalizeVector(source: IVector3): IVector3 {
	const length = Math.hypot(source.x, source.y, source.z) || 1;
	return {
		x: source.x / length,
		y: source.y / length,
		z: source.z / length,
	};
}

export function cloneColor(source: RGBA): RGBA {
	return {
		r: source.r,
		g: source.g,
		b: source.b,
		a: source.a,
	};
}
