/**
 * Creates a deep proxy that notifies a callback when any property changes.
 * Used for automatic dirty flag marking.
 */
export function makeObservable<T extends object>(
	target: T,
	onDirty: () => void
): T {
	if (typeof target !== "object" || target === null) {
		return target;
	}

	// Internal cache to avoid double proxying the same object
	const proxyMap = new WeakMap<object, object>();

	function createProxy<TObj extends object>(obj: TObj): TObj {
		const cached = proxyMap.get(obj);
		if (cached) return cached as TObj;

		const proxy = new Proxy(obj, {
			get(target, prop, receiver) {
				const value = Reflect.get(target, prop, receiver);

				// Recursively proxy nested objects
				if (
					typeof value === "object" &&
					value !== null &&
					!(value instanceof ArrayBuffer) &&
					!(value instanceof Uint8Array) &&
					!(value instanceof Uint8ClampedArray) &&
					!(value instanceof Float32Array)
				) {
					return createProxy(value);
				}

				return value;
			},

			set(target, prop, value, receiver) {
				const oldValue = Reflect.get(target, prop, receiver);

				// Only trigger if value actually changed
				if (oldValue !== value) {
					const result = Reflect.set(target, prop, value, receiver);
					if (result) {
						onDirty();
					}
					return result;
				}

				return true;
			},
		});

		proxyMap.set(obj, proxy);
		return proxy as TObj;
	}

	return createProxy(target);
}
