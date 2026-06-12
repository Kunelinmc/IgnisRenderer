import type { WarmupOptions } from "../renderers/IRenderBackend";

const DEFAULT_IDLE_TIMEOUT_MS = 16;
const DEFAULT_BACKGROUND_YIELD_INTERVAL_MS = 4;

interface WarmupHost {
	requestAnimationFrame?: (callback: FrameRequestCallback) => number;
	requestIdleCallback?: (
		callback: () => void,
		options?: { timeout?: number }
	) => number;
	setTimeout?: (callback: () => void, timeout?: number) => number;
	performance?: Pick<Performance, "now">;
}

export interface WarmupYieldController {
	yieldIfNeeded(): Promise<void>;
}

function getWarmupHost(): WarmupHost {
	return globalThis as unknown as WarmupHost;
}

function nowMs(): number {
	return getWarmupHost().performance?.now?.() ?? Date.now();
}

function waitForTimeout(): Promise<void> {
	const host = getWarmupHost();
	return new Promise((resolve) => {
		if (host.setTimeout) {
			host.setTimeout(resolve, 0);
			return;
		}
		resolve();
	});
}

function waitForAnimationFrame(): Promise<void> {
	const host = getWarmupHost();
	return new Promise((resolve) => {
		if (host.requestAnimationFrame) {
			host.requestAnimationFrame(() => {
				if (host.setTimeout) {
					host.setTimeout(resolve, 0);
				} else {
					resolve();
				}
			});
			return;
		}
		if (host.setTimeout) {
			host.setTimeout(resolve, 0);
			return;
		}
		resolve();
	});
}

function waitForIdle(): Promise<void> {
	const host = getWarmupHost();
	return new Promise((resolve) => {
		if (host.requestIdleCallback) {
			host.requestIdleCallback(() => resolve(), {
				timeout: DEFAULT_IDLE_TIMEOUT_MS,
			});
			return;
		}
		if (host.requestAnimationFrame) {
			host.requestAnimationFrame(() => {
				if (host.setTimeout) {
					host.setTimeout(resolve, 0);
				} else {
					resolve();
				}
			});
			return;
		}
		if (host.setTimeout) {
			host.setTimeout(resolve, 0);
			return;
		}
		resolve();
	});
}

export function getWarmupStartDelay(
	options: WarmupOptions = {}
): Promise<void> | null {
	switch (options.scheduling) {
		case "next-frame":
			return waitForAnimationFrame();
		case "idle":
			return waitForIdle();
		default:
			return null;
	}
}

function resolveWarmupYieldIntervalMs(options: WarmupOptions = {}): number {
	const interval = options.yieldIntervalMs;
	if (Number.isFinite(interval)) {
		return Math.max(0, interval as number);
	}
	return options.scheduling === "idle" ?
		DEFAULT_BACKGROUND_YIELD_INTERVAL_MS
	:	0;
}

function waitForWarmupYield(options: WarmupOptions): Promise<void> {
	return options.scheduling === "idle" ? waitForIdle() : waitForTimeout();
}

export function createWarmupYieldController(
	options: WarmupOptions = {}
): WarmupYieldController {
	const intervalMs = resolveWarmupYieldIntervalMs(options);
	if (intervalMs <= 0) {
		return {
			yieldIfNeeded(): Promise<void> {
				return Promise.resolve();
			},
		};
	}
	let lastYieldAt = nowMs();
	return {
		async yieldIfNeeded(): Promise<void> {
			if (nowMs() - lastYieldAt < intervalMs) {
				return;
			}
			await waitForWarmupYield(options);
			lastYieldAt = nowMs();
		},
	};
}
