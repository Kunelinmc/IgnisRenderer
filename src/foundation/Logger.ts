export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";

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

const LEVEL_ORDER: Record<Exclude<LogLevel, "silent">, number> = {
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

	public constructor(options: LoggerOptions = {}, sink: LoggerSink = console) {
		this._name = options.name ?? null;
		this._level = resolveLevel(options.level);
		this._sink = sink;
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

	private _canLog(level: Exclude<LogLevel, "silent">): boolean {
		if (this._level === "silent") return false;
		return LEVEL_ORDER[level] >= LEVEL_ORDER[this._level];
	}

	private _formatArgs(args: unknown[]): unknown[] {
		if (!this._name) return args;
		return [`[${this._name}]`, ...args];
	}
}
