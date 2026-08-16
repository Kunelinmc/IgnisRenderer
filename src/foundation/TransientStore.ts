export type TransientKey<TValue, TName extends string = string> = TName & {
	readonly __transientValueType?: TValue;
};

export function defineTransientKey<TValue, TName extends string = string>(
	name: TName
): TransientKey<TValue, TName> {
	return name as TransientKey<TValue, TName>;
}

export interface TransientStore extends Map<string, unknown> {
	get<TValue>(key: TransientKey<TValue>): TValue | undefined;
	get(key: string): unknown;
	set<TValue>(key: TransientKey<TValue>, value: TValue): this;
	set(key: string, value: unknown): this;
}

export function createTransientStore(
	entries?: Iterable<readonly [string, unknown]>
): TransientStore {
	return new Map<string, unknown>(entries) as TransientStore;
}
