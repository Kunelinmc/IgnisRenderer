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
	public debug(...args: unknown[]): void {
		if (!this._canLog("debug") || !this._sink.debug) return;
		this._sink.debug(...this._formatArgs(args));
	}

	/**
	 * Emits an info-level log.
	 */
	public info(...args: unknown[]): void {
		if (!this._canLog("info") || !this._sink.info) return;
		this._sink.info(...this._formatArgs(args));
	}

	/**
	 * Emits a warn-level log.
	 */
	public warn(...args: unknown[]): void {
		if (!this._canLog("warn") || !this._sink.warn) return;
		this._sink.warn(...this._formatArgs(args));
	}

	/**
	 * Emits an error-level log.
	 */
	public error(...args: unknown[]): void {
		if (!this._canLog("error") || !this._sink.error) return;
		this._sink.error(...this._formatArgs(args));
	}

	/**
	 * Emits a debug-level log once for the given key.
	 * Returns true if the log was emitted.
	 */
	public debugOnce(key: string, ...args: unknown[]): boolean {
		return this.logOnce("debug", key, ...args);
	}

	/**
	 * Emits an info-level log once for the given key.
	 * Returns true if the log was emitted.
	 */
	public infoOnce(key: string, ...args: unknown[]): boolean {
		return this.logOnce("info", key, ...args);
	}

	/**
	 * Emits a warn-level log once for the given key.
	 * Returns true if the log was emitted.
	 */
	public warnOnce(key: string, ...args: unknown[]): boolean {
		return this.logOnce("warn", key, ...args);
	}

	/**
	 * Emits an error-level log once for the given key.
	 * Returns true if the log was emitted.
	 */
	public errorOnce(key: string, ...args: unknown[]): boolean {
		return this.logOnce("error", key, ...args);
	}

	/**
	 * Emits a log once for the given key at the target level.
	 * Returns true if the log was emitted.
	 */
	public logOnce(
		level: EmittableLogLevel,
		key: string,
		...args: unknown[]
	): boolean {
		if (!this._canLog(level)) return false;
		const sink = this._resolveSink(level);
		if (!sink) return false;
		if (this._onceKeys.has(key)) return false;
		this._onceKeys.add(key);
		sink(...this._formatArgs(args));
		return true;
	}

	/**
	 * Returns whether the once-key has already been emitted.
	 */
	public hasOnceKey(key: string): boolean {
		return this._onceKeys.has(key);
	}

	/**
	 * Clears one once-key so it can be emitted again.
	 */
	public clearOnceKey(key: string): void {
		this._onceKeys.delete(key);
	}

	/**
	 * Clears all once-keys.
	 */
	public clearOnceKeys(): void {
		this._onceKeys.clear();
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

	private _formatArgs(args: unknown[]): unknown[] {
		if (!this._name) return args;
		return [`[${this._name}]`, ...args];
	}
}
