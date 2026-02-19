export type Listener<T extends any[] = any[]> = (...args: T) => void;

/**
 * Generic EventEmitter that supports type-safe events.
 * T should be a record where keys are event names and values are tuple arrays of arguments.
 */
export class EventEmitter<
	Events extends Record<string, any[]> = Record<string, any[]>,
> {
	private _listeners: Map<keyof Events, Listener<any>[]>;

	constructor() {
		this._listeners = new Map();
	}

	public on<K extends keyof Events>(
		event: K,
		listener: Listener<Events[K]>
	): this {
		if (!this._listeners.has(event)) {
			this._listeners.set(event, []);
		}
		this._listeners.get(event)!.push(listener as Listener<any[]>);
		return this;
	}

	public off<K extends keyof Events>(
		event: K,
		listener: Listener<Events[K]>
	): this {
		const listeners = this._listeners.get(event);
		if (!listeners) return this;

		const index = listeners.indexOf(listener as Listener<any[]>);
		if (index !== -1) {
			listeners.splice(index, 1);
		}
		return this;
	}

	public emit<K extends keyof Events>(event: K, ...args: Events[K]): boolean {
		const listeners = this._listeners.get(event);
		if (!listeners) return false;

		// Use a copy to avoid issues if listeners change during emission
		[...listeners].forEach((listener) => listener(...args));
		return true;
	}

	public once<K extends keyof Events>(
		event: K,
		listener: Listener<Events[K]>
	): this {
		const wrapper = ((...args: Events[K]) => {
			listener(...args);
			this.off(event, wrapper as any);
		}) as Listener<Events[K]>;
		return this.on(event, wrapper);
	}
}
