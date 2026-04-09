export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";
export type EmittableLogLevel = Exclude<LogLevel, "silent">;

export interface LoggerOptions {
	name?: string;
	level?: LogLevel;
}

export interface LoggerSink {
	debug?: (...args: unknown[]) => void;
	info?: (...args: unknown[]) => void;
	warn?: (...args: unknown[]) => void;
	error?: (...args: unknown[]) => void;
}

export interface LoggerEmitOptions {
	onceKey?: string | string[];
}

const LEVEL_ORDER: Record<EmittableLogLevel, number> = {
	debug: 0,
	info: 1,
	warn: 2,
	error: 3,
};

function resolveLevel(level?: LogLevel): LogLevel {
	return level ?? "info";
}

export class Logger {
	private _name: string | null;
	private _level: LogLevel;
	private _sink: LoggerSink;
	private _onceKeys: Set<string>;

	public constructor(options: LoggerOptions = {}, sink: LoggerSink = console) {
		this._name = options.name ?? null;
		this._level = resolveLevel(options.level);
		this._sink = sink;
		this._onceKeys = new Set();
	}

	/**
	 * Creates a new logger with the provided name and level.
	 */
	public static create(name: string, level: LogLevel = "info"): Logger {
		return new Logger({ name, level });
	}

	/**
	 * Updates the active logging level.
	 */
	public setLevel(level: LogLevel): void {
		this._level = level;
	}

	/**
	 * Returns the current logging level.
	 */
	public getLevel(): LogLevel {
		return this._level;
	}

	/**
	 * Emits a debug-level log.
	 */
	public debug(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return this._emit("debug", message, options);
	}

	/**
	 * Emits an info-level log.
	 */
	public info(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return this._emit("info", message, options);
	}

	/**
	 * Emits a warn-level log.
	 */
	public warn(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return this._emit("warn", message, options);
	}

	/**
	 * Emits an error-level log.
	 */
	public error(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return this._emit("error", message, options);
	}

	private _canLog(level: EmittableLogLevel): boolean {
		if (this._level === "silent") return false;
		return LEVEL_ORDER[level] >= LEVEL_ORDER[this._level];
	}

	private _resolveSink(
		level: EmittableLogLevel
	): ((...args: unknown[]) => void) | null {
		switch (level) {
			case "debug":
				return this._sink.debug ?? null;
			case "info":
				return this._sink.info ?? null;
			case "warn":
				return this._sink.warn ?? null;
			case "error":
				return this._sink.error ?? null;
			default:
				return null;
		}
	}

	private _emit(
		level: EmittableLogLevel,
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		if (!this._canLog(level)) return false;
		const sink = this._resolveSink(level);
		if (!sink) return false;
		const onceKeys = this._resolveOnceKeys(options);
		if (this._hasAnyOnceKey(onceKeys)) return false;
		sink(...this._formatArgs(this._toArgs(message)));
		this._rememberOnceKeys(onceKeys);
		return true;
	}

	private _resolveOnceKeys(options?: LoggerEmitOptions): string[] {
		const source = options?.onceKey;
		if (!source) {
			return [];
		}
		const keys = Array.isArray(source) ? source : [source];
		return keys.filter((key) => key.length > 0);
	}

	private _hasAnyOnceKey(keys: readonly string[]): boolean {
		for (const key of keys) {
			if (this._onceKeys.has(key)) {
				return true;
			}
		}
		return false;
	}

	private _rememberOnceKeys(keys: readonly string[]): void {
		for (const key of keys) {
			this._onceKeys.add(key);
		}
	}

	private _toArgs(message: string | readonly unknown[]): unknown[] {
		if (Array.isArray(message)) {
			return message.slice();
		}
		return [message];
	}

	private _formatArgs(args: unknown[]): unknown[] {
		if (!this._name) return args;
		return [`[${this._name}]`, ...args];
	}
}
