export type PlatformRuntime = "node" | "browser" | "worker" | "unknown";

export interface PlatformNavigatorGPU {
	requestAdapter?: (...args: unknown[]) => unknown;
	getPreferredCanvasFormat?: (...args: unknown[]) => unknown;
}

interface PlatformNavigatorUserAgentDataLike {
	mobile?: boolean;
}

interface PlatformNavigatorLike {
	hardwareConcurrency?: number;
	maxTouchPoints?: number;
	platform?: string;
	userAgent?: string;
	userAgentData?: PlatformNavigatorUserAgentDataLike;
	gpu?: PlatformNavigatorGPU;
}

interface PlatformProcessLike {
	versions?: {
		node?: string;
	};
}

interface PlatformDocumentLike {
	createElement?: (tagName: string) => unknown;
}

interface PlatformCanvasProbeLike {
	getContext?: (contextId: string, options?: unknown) => unknown;
}

interface PlatformScopeLike {
	process?: PlatformProcessLike;
	window?: unknown;
	document?: PlatformDocumentLike;
	navigator?: PlatformNavigatorLike;
	Worker?: unknown;
	WorkerGlobalScope?: unknown;
	SharedArrayBuffer?: unknown;
	OffscreenCanvas?: unknown;
	WebGL2RenderingContext?: unknown;
	crossOriginIsolated?: boolean;
	self?: unknown;
	importScripts?: unknown;
}

export interface PlatformFeatureSummary {
	runtime: PlatformRuntime;
	isNodeRuntime: boolean;
	isBrowserRuntime: boolean;
	isWorkerRuntime: boolean;
	hasWebGPU: boolean;
	hasWebGL2: boolean;
	hasWorker: boolean;
	hasOffscreenCanvas: boolean;
	hasSharedArrayBuffer: boolean;
	crossOriginIsolated: boolean;
	supportsSharedArrayBufferTransport: boolean;
	hardwareConcurrency: number;
}

interface PlatformDetectionCache {
	runtime?: PlatformRuntime;
	isNodeRuntime?: boolean;
	isBrowserRuntime?: boolean;
	isTouchDevice?: boolean;
	isMobileDevice?: boolean;
	isWorkerRuntime?: boolean;
	navigatorGPU?: PlatformNavigatorGPU | null;
	hasWebGPU?: boolean;
	hasWebGL2?: boolean;
	hasWorker?: boolean;
	hasOffscreenCanvas?: boolean;
	hasSharedArrayBuffer?: boolean;
	crossOriginIsolated?: boolean | null;
	hardwareConcurrency?: number | null;
}

const DEFAULT_HARDWARE_CONCURRENCY_FALLBACK = 4;
const MOBILE_USER_AGENT_PATTERN =
	/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Windows Phone|Mobile/i;

function resolveScope(scope?: unknown): PlatformScopeLike {
	return (scope ?? globalThis) as PlatformScopeLike;
}

function resolveFallbackConcurrency(fallback: number): number {
	if (!Number.isFinite(fallback)) {
		return DEFAULT_HARDWARE_CONCURRENCY_FALLBACK;
	}
	return Math.max(1, Math.floor(fallback));
}

export class Platform {
	private static readonly _detectionCache = new WeakMap<
		object,
		PlatformDetectionCache
	>();

	/**
	 * Resolves the current runtime kind.
	 */
	public static resolveRuntime(scope: unknown = globalThis): PlatformRuntime {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedRuntime = cache?.runtime;
		if (cachedRuntime !== undefined) {
			return cachedRuntime;
		}

		let runtime: PlatformRuntime = "unknown";
		if (Platform.isNodeRuntime(resolved)) {
			runtime = "node";
		} else if (Platform.isBrowserRuntime(resolved)) {
			runtime = "browser";
		} else if (Platform.isWorkerRuntime(resolved)) {
			runtime = "worker";
		}

		if (cache) {
			cache.runtime = runtime;
		}
		return runtime;
	}

	/**
	 * Returns a snapshot of common platform/runtime capabilities.
	 */
	public static detect(scope: unknown = globalThis): PlatformFeatureSummary {
		const crossOriginIsolated = Platform.isCrossOriginIsolated(scope, false);
		const hasSharedArrayBuffer = Platform.hasSharedArrayBuffer(scope);
		return {
			runtime: Platform.resolveRuntime(scope),
			isNodeRuntime: Platform.isNodeRuntime(scope),
			isBrowserRuntime: Platform.isBrowserRuntime(scope),
			isWorkerRuntime: Platform.isWorkerRuntime(scope),
			hasWebGPU: Platform.hasWebGPU(scope),
			hasWebGL2: Platform.hasWebGL2(scope),
			hasWorker: Platform.hasWorker(scope),
			hasOffscreenCanvas: Platform.hasOffscreenCanvas(scope),
			hasSharedArrayBuffer,
			crossOriginIsolated,
			supportsSharedArrayBufferTransport: hasSharedArrayBuffer &&
				crossOriginIsolated,
			hardwareConcurrency: Platform.getHardwareConcurrency(
				DEFAULT_HARDWARE_CONCURRENCY_FALLBACK,
				scope
			),
		};
	}

	/**
	 * Returns true when running in Node.js.
	 */
	public static isNodeRuntime(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.isNodeRuntime;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const nodeVersion = resolved.process?.versions?.node;
		const detected = typeof nodeVersion === "string" && nodeVersion.length > 0;
		if (cache) {
			cache.isNodeRuntime = detected;
		}
		return detected;
	}

	/**
	 * Returns true when running in a browser window context.
	 */
	public static isBrowserRuntime(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.isBrowserRuntime;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const detected = !!resolved.window &&
			typeof resolved.window === "object" &&
			typeof resolved.document?.createElement === "function";
		if (cache) {
			cache.isBrowserRuntime = detected;
		}
		return detected;
	}

	/**
	 * Returns true when touch input capability appears to be available.
	 */
	public static isTouchDevice(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.isTouchDevice;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const maxTouchPoints = resolved.navigator?.maxTouchPoints;
		if (
			typeof maxTouchPoints === "number" &&
			Number.isFinite(maxTouchPoints) &&
			maxTouchPoints > 0
		) {
			if (cache) {
				cache.isTouchDevice = true;
			}
			return true;
		}

		const windowValue = resolved.window;
		const detected = !!windowValue && typeof windowValue === "object" &&
			"ontouchstart" in windowValue;
		if (cache) {
			cache.isTouchDevice = detected;
		}
		return detected;
	}

	/**
	 * Returns true when the current client appears to be a mobile device.
	 */
	public static isMobileDevice(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.isMobileDevice;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const navigatorValue = resolved.navigator;
		const mobileFromUserAgentData = navigatorValue?.userAgentData?.mobile;
		if (typeof mobileFromUserAgentData === "boolean") {
			if (cache) {
				cache.isMobileDevice = mobileFromUserAgentData;
			}
			return mobileFromUserAgentData;
		}

		const userAgent = navigatorValue?.userAgent;
		if (
			typeof userAgent === "string" &&
			userAgent.length > 0 &&
			MOBILE_USER_AGENT_PATTERN.test(userAgent)
		) {
			if (cache) {
				cache.isMobileDevice = true;
			}
			return true;
		}

		const platform = navigatorValue?.platform;
		const detected = platform === "MacIntel" &&
			Platform.isTouchDevice(resolved);
		if (cache) {
			cache.isMobileDevice = detected;
		}
		return detected;
	}

	/**
	 * Returns true when running in a worker global scope.
	 */
	public static isWorkerRuntime(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.isWorkerRuntime;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		if (Platform.isNodeRuntime(resolved)) {
			if (cache) {
				cache.isWorkerRuntime = false;
			}
			return false;
		}

		const selfValue = resolved.self;
		const workerGlobalScopeCtor = resolved.WorkerGlobalScope;
		if (
			selfValue &&
			typeof selfValue === "object" &&
			typeof workerGlobalScopeCtor === "function"
		) {
			try {
				if (
					selfValue instanceof
					(workerGlobalScopeCtor as new (...args: never[]) => object)
				) {
					if (cache) {
						cache.isWorkerRuntime = true;
					}
					return true;
				}
			} catch (_error) {
				// Ignore cross-realm or constructor edge cases.
			}
		}

		const detected = (
			selfValue === resolved && typeof resolved.importScripts === "function"
		);
		if (cache) {
			cache.isWorkerRuntime = detected;
		}
		return detected;
	}

	/**
	 * Returns the navigator.gpu object when available.
	 */
	public static getNavigatorGPU(
		scope: unknown = globalThis
	): PlatformNavigatorGPU | null {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.navigatorGPU;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const navigatorValue = resolved.navigator;
		const gpu = navigatorValue?.gpu;
		const detected = (!gpu || typeof gpu !== "object")
			? null
			: gpu;
		if (cache) {
			cache.navigatorGPU = detected;
		}
		return detected;
	}

	/**
	 * Returns true when WebGPU appears to be available.
	 */
	public static hasWebGPU(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hasWebGPU;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const gpu = Platform.getNavigatorGPU(scope);
		const detected = !!gpu && typeof gpu.requestAdapter === "function";
		if (cache) {
			cache.hasWebGPU = detected;
		}
		return detected;
	}

	/**
	 * Returns true when WebGL2 context creation appears to be available.
	 */
	public static hasWebGL2(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hasWebGL2;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		let detected = false;
		if (
			Platform._canvasSupportsWebGL2(
				Platform._createOffscreenCanvasProbe(resolved)
			)
		) {
			detected = true;
		}

		if (
			!detected &&
			Platform._canvasSupportsWebGL2(
				Platform._createDocumentCanvasProbe(resolved)
			)
		) {
			detected = true;
		}

		if (!detected) {
			detected = typeof resolved.WebGL2RenderingContext === "function";
		}

		if (cache) {
			cache.hasWebGL2 = detected;
		}
		return detected;
	}

	/**
	 * Returns true when Worker constructor is available.
	 */
	public static hasWorker(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hasWorker;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const detected = typeof resolved.Worker === "function";
		if (cache) {
			cache.hasWorker = detected;
		}
		return detected;
	}

	/**
	 * Returns true when OffscreenCanvas constructor is available.
	 */
	public static hasOffscreenCanvas(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hasOffscreenCanvas;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const detected = typeof resolved.OffscreenCanvas === "function";
		if (cache) {
			cache.hasOffscreenCanvas = detected;
		}
		return detected;
	}

	/**
	 * Returns true when SharedArrayBuffer constructor is available.
	 */
	public static hasSharedArrayBuffer(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hasSharedArrayBuffer;
		if (cachedValue !== undefined) {
			return cachedValue;
		}

		const detected = typeof resolved.SharedArrayBuffer === "function";
		if (cache) {
			cache.hasSharedArrayBuffer = detected;
		}
		return detected;
	}

	/**
	 * Returns whether the runtime is cross-origin isolated.
	 */
	public static isCrossOriginIsolated(
		scope: unknown = globalThis,
		unknownValue: boolean = false
	): boolean {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.crossOriginIsolated;
		if (cachedValue !== undefined) {
			return cachedValue === null ? unknownValue : cachedValue;
		}

		const value = resolved.crossOriginIsolated;
		const detected = typeof value === "boolean" ? value : null;
		if (cache) {
			cache.crossOriginIsolated = detected;
		}
		return detected === null ? unknownValue : detected;
	}

	/**
	 * Returns true when SharedArrayBuffer transport requirements are met.
	 */
	public static supportsSharedArrayBufferTransport(
		scope: unknown = globalThis,
		unknownCrossOriginIsolatedValue: boolean = false
	): boolean {
		return (
			Platform.hasSharedArrayBuffer(scope) &&
			Platform.isCrossOriginIsolated(scope, unknownCrossOriginIsolatedValue)
		);
	}

	/**
	 * Returns navigator.hardwareConcurrency clamped to a safe integer.
	 */
	public static getHardwareConcurrency(
		fallback: number = DEFAULT_HARDWARE_CONCURRENCY_FALLBACK,
		scope: unknown = globalThis
	): number {
		const resolved = resolveScope(scope);
		const cache = Platform._getDetectionCache(resolved);
		const cachedValue = cache?.hardwareConcurrency;
		if (cachedValue !== undefined) {
			return cachedValue === null
				? resolveFallbackConcurrency(fallback)
				: cachedValue;
		}

		const navigatorValue = resolved.navigator?.hardwareConcurrency;
		const detected = Number.isFinite(navigatorValue)
			? Math.max(1, Math.floor(navigatorValue as number))
			: null;
		if (cache) {
			cache.hardwareConcurrency = detected;
		}
		return detected === null
			? resolveFallbackConcurrency(fallback)
			: detected;
	}

	private static _getDetectionCache(
		scope: PlatformScopeLike
	): PlatformDetectionCache | null {
		if (!scope || (typeof scope !== "object" && typeof scope !== "function")) {
			return null;
		}

		let cached = Platform._detectionCache.get(scope as object);
		if (!cached) {
			cached = {};
			Platform._detectionCache.set(scope as object, cached);
		}
		return cached;
	}

	private static _canvasSupportsWebGL2(
		canvas: PlatformCanvasProbeLike | null
	): boolean {
		if (!canvas || typeof canvas.getContext !== "function") return false;
		try {
			return !!canvas.getContext("webgl2");
		} catch (_error) {
			return false;
		}
	}

	private static _createOffscreenCanvasProbe(
		scope: unknown
	): PlatformCanvasProbeLike | null {
		const resolved = resolveScope(scope);
		const offscreenCanvasCtor = resolved.OffscreenCanvas;
		if (typeof offscreenCanvasCtor === "function") {
			try {
				const canvas = new (offscreenCanvasCtor as new (
					width: number,
					height: number
				) => PlatformCanvasProbeLike)(1, 1);
				if (typeof canvas.getContext === "function") {
					return canvas;
				}
			} catch (_error) {
				// Ignore constructor failures.
			}
		}
		return null;
	}

	private static _createDocumentCanvasProbe(
		scope: unknown
	): PlatformCanvasProbeLike | null {
		const resolved = resolveScope(scope);
		const documentValue = resolved.document;
		if (!documentValue || typeof documentValue.createElement !== "function") {
			return null;
		}
		try {
			const canvas = documentValue.createElement(
				"canvas"
			) as PlatformCanvasProbeLike;
			if (typeof canvas.getContext === "function") {
				return canvas;
			}
		} catch (_error) {
			return null;
		}

		return null;
	}
}
