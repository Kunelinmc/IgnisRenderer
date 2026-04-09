export type LogLevel = "debug" | "info" | "warn" | "error" | "silent";
export type EmittableLogLevel = Exclude<LogLevel, "silent">;

export interface LoggerConfigureOptions {
	level?: LogLevel;
	sink?: LoggerSink;
	resetOnceKeys?: boolean;
}

export interface LoggerSink {
	debug?: (...args: unknown[]) => void;
	info?: (...args: unknown[]) => void;
	warn?: (...args: unknown[]) => void;
	error?: (...args: unknown[]) => void;
}

export interface LoggerEmitOptions {
	scope?: string;
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
	private static _level: LogLevel = "info";
	private static _sink: LoggerSink = console;
	private static _onceKeys = new Set<string>();

	private constructor() {}

	/**
	 * Configures global logger behavior.
	 */
	public static configure(options: LoggerConfigureOptions = {}): void {
		if (options.level) {
			Logger._level = resolveLevel(options.level);
		}
		if (options.sink) {
			Logger._sink = options.sink;
		}
		if (options.resetOnceKeys) {
			Logger._onceKeys.clear();
		}
	}

	/**
	 * Resets logger configuration to defaults.
	 */
	public static reset(): void {
		Logger._level = "info";
		Logger._sink = console;
		Logger._onceKeys.clear();
	}

	/**
	 * Updates the active logging level.
	 */
	public static setLevel(level: LogLevel): void {
		Logger._level = resolveLevel(level);
	}

	/**
	 * Returns the current logging level.
	 */
	public static getLevel(): LogLevel {
		return Logger._level;
	}

	/**
	 * Updates the active sink.
	 */
	public static setSink(sink: LoggerSink): void {
		Logger._sink = sink;
	}

	/**
	 * Emits a debug-level log.
	 */
	public static debug(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return Logger._emit("debug", message, options);
	}

	/**
	 * Emits an info-level log.
	 */
	public static info(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return Logger._emit("info", message, options);
	}

	/**
	 * Emits a warn-level log.
	 */
	public static warn(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return Logger._emit("warn", message, options);
	}

	/**
	 * Emits an error-level log.
	 */
	public static error(
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		return Logger._emit("error", message, options);
	}

	private static _canLog(level: EmittableLogLevel): boolean {
		if (Logger._level === "silent") return false;
		return LEVEL_ORDER[level] >= LEVEL_ORDER[Logger._level];
	}

	private static _resolveSink(
		level: EmittableLogLevel
	): ((...args: unknown[]) => void) | null {
		switch (level) {
			case "debug":
				return Logger._sink.debug ?? null;
			case "info":
				return Logger._sink.info ?? null;
			case "warn":
				return Logger._sink.warn ?? null;
			case "error":
				return Logger._sink.error ?? null;
			default:
				return null;
		}
	}

	private static _emit(
		level: EmittableLogLevel,
		message: string | readonly unknown[],
		options?: LoggerEmitOptions
	): boolean {
		if (!Logger._canLog(level)) return false;
		const sink = Logger._resolveSink(level);
		if (!sink) return false;
		const onceKeys = Logger._resolveOnceKeys(options);
		if (Logger._hasAnyOnceKey(onceKeys)) return false;
		sink(...Logger._formatArgs(Logger._toArgs(message), options));
		Logger._rememberOnceKeys(onceKeys);
		return true;
	}

	private static _resolveOnceKeys(options?: LoggerEmitOptions): string[] {
		const source = options?.onceKey;
		if (!source) {
			return [];
		}
		const keys = Array.isArray(source) ? source : [source];
		return keys.filter((key) => key.length > 0);
	}

	private static _hasAnyOnceKey(keys: readonly string[]): boolean {
		for (const key of keys) {
			if (Logger._onceKeys.has(key)) {
				return true;
			}
		}
		return false;
	}

	private static _rememberOnceKeys(keys: readonly string[]): void {
		for (const key of keys) {
			Logger._onceKeys.add(key);
		}
	}

	private static _toArgs(message: string | readonly unknown[]): unknown[] {
		if (Array.isArray(message)) {
			return message.slice();
		}
		return [message];
	}

	private static _formatArgs(
		args: unknown[],
		options?: LoggerEmitOptions
	): unknown[] {
		const scope = options?.scope?.trim();
		if (!scope) return args;
		return [`[${scope}]`, ...args];
	}
}

export type LoggerStatic = typeof Logger;
