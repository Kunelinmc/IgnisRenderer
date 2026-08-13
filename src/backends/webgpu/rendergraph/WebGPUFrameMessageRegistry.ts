import {
	WebGPUFrameMessageDispatchError,
	type WebGPUFrameMessageFailure,
} from "../../../foundation/Error";

import {
	WEBGPU_FRAME_MESSAGE_PHASES,
	WebGPUFrameMessageSnapshot,
	type WebGPUFrameMessageDescriptor,
	type WebGPUFrameMessageHandler,
	type WebGPUFrameMessagePhase,
	type WebGPUFrameMessagePublisher,
	type WebGPUFrameMessageReader,
} from "./WebGPUFrameMessage";

interface WebGPUFrameMessagePublication {
	readonly descriptor: WebGPUFrameMessageDescriptor<unknown>;
	readonly value: unknown;
	readonly index: number;
}

interface WebGPUFrameMessageSchedule {
	readonly waves: readonly (readonly WebGPUFrameMessageHandler[])[];
}

/**
 * Backend-private typed asynchronous frame-message scheduler.
 *
 * @internal Owned by the WebGPU frame runtime composition root.
 */
export class WebGPUFrameMessageRegistry {
	private readonly _handlers: WebGPUFrameMessageHandler[] = [];
	private readonly _descriptors = new Map<
		string,
		WebGPUFrameMessageDescriptor<unknown>
	>();
	private readonly _schedules = new Map<
		WebGPUFrameMessagePhase,
		WebGPUFrameMessageSchedule
	>();
	private _sealed = false;

	public register(handler: WebGPUFrameMessageHandler): void {
		if (this._sealed) {
			throw new Error("WebGPU frame message registry is sealed.");
		}
		if (!handler.id || !handler.moduleId) {
			throw new Error("WebGPU frame message handlers require id and moduleId.");
		}
		if (this._handlers.some((entry) =>
			entry.moduleId === handler.moduleId && entry.id === handler.id)) {
			throw new Error(
				`WebGPU frame message handler "${handler.moduleId}:${handler.id}" is duplicate.`,
			);
		}
		for (const input of handler.inputs ?? []) {
			this._registerDescriptor(input.descriptor);
		}
		for (const output of handler.outputs ?? []) {
			this._registerDescriptor(output);
		}
		this._handlers.push(handler);
	}

	public seal(): void {
		if (this._sealed) return;
		this._validateProducers();
		for (const phase of WEBGPU_FRAME_MESSAGE_PHASES) {
			this._schedules.set(phase, this._buildSchedule(phase));
		}
		this._sealed = true;
	}

	public async dispatch(
		phase: WebGPUFrameMessagePhase,
		options: {
			readonly prior?: WebGPUFrameMessageSnapshot;
			readonly seeds?: readonly {
				readonly descriptor: WebGPUFrameMessageDescriptor<unknown>;
				readonly value: unknown;
			}[];
		} = {},
	): Promise<WebGPUFrameMessageSnapshot> {
		this._assertSealed();
		const committed = options.prior?.cloneValues() ?? new Map();
		const provisional = new Map(committed);
		for (const seed of options.seeds ?? []) {
			this._assertKnownDescriptor(seed.descriptor);
			if (!seed.descriptor.seeded || seed.descriptor.phase !== phase) {
				throw new Error(
					`WebGPU frame message "${seed.descriptor.id}" is not a ${phase} seed.`,
				);
			}
			this._appendValue(provisional, seed.descriptor, seed.value);
		}
		const schedule = this._schedules.get(phase)!;
		for (let waveIndex = 0; waveIndex < schedule.waves.length; waveIndex++) {
			const wave = schedule.waves[waveIndex];
			const visible = new WebGPUFrameMessageSnapshot(provisional);
			const executions = wave.map(async (handler) => {
				const publications: WebGPUFrameMessagePublication[] = [];
				const declaredInputs = new Set(
					(handler.inputs ?? []).map((input) => input.descriptor),
				);
				const declaredOutputs = new Set(handler.outputs ?? []);
				const reader = this._createReader(visible, declaredInputs, handler);
				const publisher: WebGPUFrameMessagePublisher = {
					publish: (descriptor, value) => {
						if (!declaredOutputs.has(descriptor)) {
							throw new Error(
								`WebGPU frame message handler "${handler.moduleId}:${handler.id}" ` +
								`cannot publish undeclared output "${descriptor.id}".`,
							);
						}
						publications.push({
							descriptor,
							value,
							index: publications.length,
						});
					},
				};
				await handler.run(reader, publisher);
				this._validateHandlerPublications(handler, publications);
				return { handler, publications };
			});
			const settled = await Promise.allSettled(executions);
			const failures: WebGPUFrameMessageFailure[] = [];
			for (let index = 0; index < settled.length; index++) {
				const result = settled[index];
				if (result.status === "rejected") {
					const handler = wave[index];
					failures.push({
						phase,
						wave: waveIndex,
						moduleId: handler.moduleId,
						handlerId: handler.id,
						cause: result.reason,
					});
				}
			}
			if (failures.length > 0) {
				failures.sort(compareFailures);
				throw new WebGPUFrameMessageDispatchError(failures);
			}
			const completed = settled
				.map((result) => result.status === "fulfilled" ? result.value : null)
				.filter((entry): entry is Awaited<typeof executions[number]> => !!entry)
				.sort((a, b) => compareHandlers(a.handler, b.handler));
			for (const entry of completed) {
				for (const publication of entry.publications) {
					this._appendValue(
						provisional,
						publication.descriptor,
						publication.value,
					);
				}
			}
		}
		this._validatePhaseCardinality(phase, provisional);
		return new WebGPUFrameMessageSnapshot(provisional);
	}

	private _registerDescriptor(
		descriptor: WebGPUFrameMessageDescriptor<unknown>,
	): void {
		const existing = this._descriptors.get(descriptor.id);
		if (existing && existing !== descriptor) {
			throw new Error(
				`WebGPU frame message descriptor "${descriptor.id}" has conflicting identities.`,
			);
		}
		this._descriptors.set(descriptor.id, descriptor);
	}

	private _validateProducers(): void {
		const producers = this._collectProducers();
		for (const descriptor of this._descriptors.values()) {
			const entries = producers.get(descriptor) ?? [];
			if (!descriptor.seeded && entries.length === 0) {
				throw new Error(
					`WebGPU frame message "${descriptor.id}" has no producer.`,
				);
			}
			if (descriptor.cardinality === "one" && entries.length > 1) {
				throw new Error(
					`WebGPU frame message "${descriptor.id}" has duplicate single-value producers.`,
				);
			}
		}
	}

	private _buildSchedule(phase: WebGPUFrameMessagePhase): WebGPUFrameMessageSchedule {
		const handlers = this._handlers.filter((handler) => handler.phase === phase);
		const producers = this._collectProducers();
		const dependencies = new Map<WebGPUFrameMessageHandler, Set<WebGPUFrameMessageHandler>>();
		for (const handler of handlers) {
			const required = new Set<WebGPUFrameMessageHandler>();
			for (const input of handler.inputs ?? []) {
				if (phaseIndex(input.descriptor.phase) > phaseIndex(handler.phase)) {
					throw new Error(
						`WebGPU frame message handler "${handler.moduleId}:${handler.id}" ` +
						`depends on later phase message "${input.descriptor.id}".`,
					);
				}
				if (input.descriptor.phase !== phase) continue;
				for (const producer of producers.get(input.descriptor) ?? []) {
					if (producer !== handler) required.add(producer);
					else {
						throw new Error(
							`WebGPU frame message handler "${handler.moduleId}:${handler.id}" ` +
							`has a self-cycle through "${input.descriptor.id}".`,
						);
					}
				}
			}
			dependencies.set(handler, required);
		}
		const remaining = new Set(handlers);
		const waves: WebGPUFrameMessageHandler[][] = [];
		while (remaining.size > 0) {
			const wave = [...remaining]
				.filter((handler) => [...dependencies.get(handler)!]
					.every((dependency) => !remaining.has(dependency)))
				.sort(compareHandlers);
			if (wave.length === 0) {
				throw new Error(`WebGPU frame message ${phase} handler graph contains a cycle.`);
			}
			waves.push(wave);
			for (const handler of wave) remaining.delete(handler);
		}
		return { waves };
	}

	private _collectProducers(): Map<
		WebGPUFrameMessageDescriptor<unknown>,
		WebGPUFrameMessageHandler[]
	> {
		const result = new Map<
			WebGPUFrameMessageDescriptor<unknown>,
			WebGPUFrameMessageHandler[]
		>();
		for (const handler of this._handlers) {
			for (const descriptor of handler.outputs ?? []) {
				if (descriptor.phase !== handler.phase) {
					throw new Error(
						`WebGPU frame message handler "${handler.moduleId}:${handler.id}" ` +
						`publishes "${descriptor.id}" in another phase.`,
					);
				}
				const entries = result.get(descriptor) ?? [];
				entries.push(handler);
				result.set(descriptor, entries);
			}
		}
		return result;
	}

	private _createReader(
		snapshot: WebGPUFrameMessageSnapshot,
		declared: ReadonlySet<WebGPUFrameMessageDescriptor<unknown>>,
		handler: WebGPUFrameMessageHandler,
	): WebGPUFrameMessageReader {
		const assertDeclared = (descriptor: WebGPUFrameMessageDescriptor<unknown>) => {
			if (!declared.has(descriptor)) {
				throw new Error(
					`WebGPU frame message handler "${handler.moduleId}:${handler.id}" ` +
					`cannot read undeclared input "${descriptor.id}".`,
				);
			}
		};
		return {
			get: (descriptor) => {
				assertDeclared(descriptor);
				return snapshot.get(descriptor);
			},
			getOptional: (descriptor) => {
				assertDeclared(descriptor);
				return snapshot.getOptional(descriptor);
			},
			getAll: (descriptor) => {
				assertDeclared(descriptor);
				return snapshot.getAll(descriptor);
			},
		};
	}

	private _validateHandlerPublications(
		handler: WebGPUFrameMessageHandler,
		publications: readonly WebGPUFrameMessagePublication[],
	): void {
		for (const descriptor of handler.outputs ?? []) {
			if (descriptor.cardinality !== "one") continue;
			const count = publications.filter((entry) => entry.descriptor === descriptor).length;
			if (count !== 1) {
				throw new Error(
					`WebGPU frame message handler "${handler.moduleId}:${handler.id}" must ` +
					`publish "${descriptor.id}" exactly once.`,
				);
			}
		}
	}

	private _validatePhaseCardinality(
		phase: WebGPUFrameMessagePhase,
		values: ReadonlyMap<WebGPUFrameMessageDescriptor<unknown>, readonly unknown[]>,
	): void {
		for (const descriptor of this._descriptors.values()) {
			if (descriptor.phase !== phase || descriptor.cardinality !== "one") continue;
			const count = values.get(descriptor)?.length ?? 0;
			if (count !== 1) {
				throw new Error(
					`WebGPU frame message "${descriptor.id}" requires exactly one value.`,
				);
			}
		}
	}

	private _appendValue(
		values: Map<WebGPUFrameMessageDescriptor<unknown>, readonly unknown[]>,
		descriptor: WebGPUFrameMessageDescriptor<unknown>,
		value: unknown,
	): void {
		const existing = values.get(descriptor) ?? [];
		if (descriptor.cardinality === "one" && existing.length > 0) {
			throw new Error(`WebGPU frame message "${descriptor.id}" is already published.`);
		}
		values.set(descriptor, Object.freeze([...existing, value]));
	}

	private _assertKnownDescriptor(
		descriptor: WebGPUFrameMessageDescriptor<unknown>,
	): void {
		if (this._descriptors.get(descriptor.id) !== descriptor) {
			throw new Error(`WebGPU frame message "${descriptor.id}" is not registered.`);
		}
	}

	private _assertSealed(): void {
		if (!this._sealed) {
			throw new Error("WebGPU frame message registry is not sealed.");
		}
	}
}

function phaseIndex(phase: WebGPUFrameMessagePhase): number {
	return WEBGPU_FRAME_MESSAGE_PHASES.indexOf(phase);
}

function compareHandlers(
	a: WebGPUFrameMessageHandler,
	b: WebGPUFrameMessageHandler,
): number {
	return a.moduleId.localeCompare(b.moduleId) || a.id.localeCompare(b.id);
}

function compareFailures(
	a: WebGPUFrameMessageFailure,
	b: WebGPUFrameMessageFailure,
): number {
	return a.moduleId.localeCompare(b.moduleId) || a.handlerId.localeCompare(b.handlerId);
}
