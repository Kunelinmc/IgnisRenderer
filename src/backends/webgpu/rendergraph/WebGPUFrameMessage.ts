export const WEBGPU_FRAME_MESSAGE_PHASES = [
	"analysis",
	"configuration",
	"planning",
] as const;

export type WebGPUFrameMessagePhase =
	(typeof WEBGPU_FRAME_MESSAGE_PHASES)[number];

export type WebGPUFrameMessageCardinality = "one" | "many";

/** @internal Canonical identity and ownership for one frame-message topic. */
export interface WebGPUFrameMessageDescriptor<TValue> {
	readonly id: string;
	readonly ownerId: string;
	readonly phase: WebGPUFrameMessagePhase;
	readonly cardinality: WebGPUFrameMessageCardinality;
	readonly seeded: boolean;
	readonly __valueType?: TValue;
}

/** @internal Defines one backend-private frame-message topic. */
export function defineWebGPUFrameMessage<TValue>(options: {
	readonly id: string;
	readonly ownerId: string;
	readonly phase: WebGPUFrameMessagePhase;
	readonly cardinality?: WebGPUFrameMessageCardinality;
	readonly seeded?: boolean;
}): WebGPUFrameMessageDescriptor<TValue> {
	if (!options.id || !options.ownerId) {
		throw new Error("WebGPU frame message descriptors require id and ownerId.");
	}
	return Object.freeze({
		id: options.id,
		ownerId: options.ownerId,
		phase: options.phase,
		cardinality: options.cardinality ?? "one",
		seeded: options.seeded === true,
	});
}

export interface WebGPUFrameMessageInput<TValue = unknown> {
	readonly descriptor: WebGPUFrameMessageDescriptor<TValue>;
	readonly required?: boolean;
}

export interface WebGPUFrameMessageReader {
	get<TValue>(descriptor: WebGPUFrameMessageDescriptor<TValue>): TValue;
	getOptional<TValue>(
		descriptor: WebGPUFrameMessageDescriptor<TValue>,
	): TValue | undefined;
	getAll<TValue>(
		descriptor: WebGPUFrameMessageDescriptor<TValue>,
	): readonly TValue[];
}

export interface WebGPUFrameMessagePublisher {
	publish<TValue>(
		descriptor: WebGPUFrameMessageDescriptor<TValue>,
		value: TValue,
	): void;
}

export interface WebGPUFrameMessageHandler {
	readonly id: string;
	readonly moduleId: string;
	readonly phase: WebGPUFrameMessagePhase;
	readonly inputs?: readonly WebGPUFrameMessageInput[];
	readonly outputs?: readonly WebGPUFrameMessageDescriptor<unknown>[];
	run(
		messages: WebGPUFrameMessageReader,
		publisher: WebGPUFrameMessagePublisher,
	): void | Promise<void>;
}

/** Immutable messages committed through the latest completed phase. */
export class WebGPUFrameMessageSnapshot implements WebGPUFrameMessageReader {
	private readonly _values: ReadonlyMap<
		WebGPUFrameMessageDescriptor<unknown>,
		readonly unknown[]
	>;

	/** @internal Created only by `WebGPUFrameMessageRegistry`. */
	public constructor(
		values?: ReadonlyMap<
			WebGPUFrameMessageDescriptor<unknown>,
			readonly unknown[]
		>,
	) {
		this._values = values ?? new Map();
	}

	public get<TValue>(descriptor: WebGPUFrameMessageDescriptor<TValue>): TValue {
		const values = this._values.get(descriptor);
		if (!values || values.length !== 1) {
			throw new Error(`WebGPU frame message "${descriptor.id}" is unavailable.`);
		}
		return values[0] as TValue;
	}

	public getOptional<TValue>(
		descriptor: WebGPUFrameMessageDescriptor<TValue>,
	): TValue | undefined {
		const values = this._values.get(descriptor);
		if (!values || values.length === 0) return undefined;
		if (values.length !== 1) {
			throw new Error(`WebGPU frame message "${descriptor.id}" is not singular.`);
		}
		return values[0] as TValue;
	}

	public getAll<TValue>(
		descriptor: WebGPUFrameMessageDescriptor<TValue>,
	): readonly TValue[] {
		return (this._values.get(descriptor) ?? []) as readonly TValue[];
	}

	/** @internal Supplies committed values to the next phase transaction. */
	public cloneValues(): Map<WebGPUFrameMessageDescriptor<unknown>, readonly unknown[]> {
		return new Map(this._values);
	}
}
