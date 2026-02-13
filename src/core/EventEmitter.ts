export type Listener = (...args: unknown[]) => void;

export class EventEmitter {
	private _listeners: Map<string, Listener[]>;

	constructor() {
		this._listeners = new Map();
	}

	public on(event: string, listener: Listener): this {
		if (!this._listeners.has(event)) {
			this._listeners.set(event, []);
		}
		this._listeners.get(event)!.push(listener);
		return this;
	}

	public off(event: string, listener: Listener): this {
		const listeners = this._listeners.get(event);
		if (!listeners) return this;

		const index = listeners.indexOf(listener);
		if (index !== -1) {
			listeners.splice(index, 1);
		}
		return this;
	}

	public emit(event: string, ...args: unknown[]): boolean {
		const listeners = this._listeners.get(event);
		if (!listeners) return false;

		// Use a copy to avoid issues if listeners change during emission
		[...listeners].forEach((listener) => listener(...args));
		return true;
	}

	public once(event: string, listener: Listener): this {
		const wrapper = (...args: unknown[]) => {
			listener(...args);
			this.off(event, wrapper);
		};
		return this.on(event, wrapper);
	}
}
