type BuildTuple<
	T,
	N extends number,
	Result extends T[] = [],
> = Result["length"] extends N
	? Result
	: BuildTuple<T, N, [...Result, T]>;

/** A mutable array with exactly `N` elements at assignment boundaries. */
export type Tuple<T, N extends number> = N extends N
	? number extends N
		? T[]
		: BuildTuple<T, N>
	: never;
