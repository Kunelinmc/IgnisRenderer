export type PlatformRuntime = "node" | "browser" | "worker" | "unknown";

export interface PlatformNavigatorGPU {
	requestAdapter?: (...args: unknown[]) => unknown;
	getPreferredCanvasFormat?: (...args: unknown[]) => unknown;
}

interface PlatformNavigatorLike {
	hardwareConcurrency?: number;
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

const DEFAULT_HARDWARE_CONCURRENCY_FALLBACK = 4;

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
	/**
	 * Resolves the current runtime kind.
	 */
	public static resolveRuntime(scope: unknown = globalThis): PlatformRuntime {
		if (Platform.isNodeRuntime(scope)) return "node";
		if (Platform.isBrowserRuntime(scope)) return "browser";
		if (Platform.isWorkerRuntime(scope)) return "worker";
		return "unknown";
	}

	/**
	 * Returns a snapshot of common platform/runtime capabilities.
	 */
	public static detect(scope: unknown = globalThis): PlatformFeatureSummary {
		const crossOriginIsolated = Platform.isCrossOriginIsolated(scope, false);
		return {
			runtime: Platform.resolveRuntime(scope),
			isNodeRuntime: Platform.isNodeRuntime(scope),
			isBrowserRuntime: Platform.isBrowserRuntime(scope),
			isWorkerRuntime: Platform.isWorkerRuntime(scope),
			hasWebGPU: Platform.hasWebGPU(scope),
			hasWebGL2: Platform.hasWebGL2(scope),
			hasWorker: Platform.hasWorker(scope),
			hasOffscreenCanvas: Platform.hasOffscreenCanvas(scope),
			hasSharedArrayBuffer: Platform.hasSharedArrayBuffer(scope),
			crossOriginIsolated,
			supportsSharedArrayBufferTransport:
				Platform.hasSharedArrayBuffer(scope) && crossOriginIsolated,
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
		const nodeVersion = resolveScope(scope).process?.versions?.node;
		return typeof nodeVersion === "string" && nodeVersion.length > 0;
	}

	/**
	 * Returns true when running in a browser window context.
	 */
	public static isBrowserRuntime(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		if (!resolved.window || typeof resolved.window !== "object") return false;
		return typeof resolved.document?.createElement === "function";
	}

	/**
	 * Returns true when running in a worker global scope.
	 */
	public static isWorkerRuntime(scope: unknown = globalThis): boolean {
		const resolved = resolveScope(scope);
		if (Platform.isNodeRuntime(resolved)) return false;

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
					return true;
				}
			} catch (_error) {
				// Ignore cross-realm or constructor edge cases.
			}
		}

		return (
			selfValue === resolved && typeof resolved.importScripts === "function"
		);
	}

	/**
	 * Returns the navigator.gpu object when available.
	 */
	public static getNavigatorGPU(
		scope: unknown = globalThis
	): PlatformNavigatorGPU | null {
		const navigatorValue = resolveScope(scope).navigator;
		const gpu = navigatorValue?.gpu;
		if (!gpu || typeof gpu !== "object") return null;
		return gpu;
	}

	/**
	 * Returns true when WebGPU appears to be available.
	 */
	public static hasWebGPU(scope: unknown = globalThis): boolean {
		const gpu = Platform.getNavigatorGPU(scope);
		return !!gpu && typeof gpu.requestAdapter === "function";
	}

	/**
	 * Returns true when WebGL2 context creation appears to be available.
	 */
	public static hasWebGL2(scope: unknown = globalThis): boolean {
		if (
			Platform._canvasSupportsWebGL2(
				Platform._createOffscreenCanvasProbe(scope)
			)
		) {
			return true;
		}
		if (
			Platform._canvasSupportsWebGL2(Platform._createDocumentCanvasProbe(scope))
		) {
			return true;
		}
		return typeof resolveScope(scope).WebGL2RenderingContext === "function";
	}

	/**
	 * Returns true when Worker constructor is available.
	 */
	public static hasWorker(scope: unknown = globalThis): boolean {
		return typeof resolveScope(scope).Worker === "function";
	}

	/**
	 * Returns true when OffscreenCanvas constructor is available.
	 */
	public static hasOffscreenCanvas(scope: unknown = globalThis): boolean {
		return typeof resolveScope(scope).OffscreenCanvas === "function";
	}

	/**
	 * Returns true when SharedArrayBuffer constructor is available.
	 */
	public static hasSharedArrayBuffer(scope: unknown = globalThis): boolean {
		return typeof resolveScope(scope).SharedArrayBuffer === "function";
	}

	/**
	 * Returns whether the runtime is cross-origin isolated.
	 */
	public static isCrossOriginIsolated(
		scope: unknown = globalThis,
		unknownValue: boolean = false
	): boolean {
		const value = resolveScope(scope).crossOriginIsolated;
		if (typeof value === "boolean") return value;
		return unknownValue;
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
		const navigatorValue = resolveScope(scope).navigator?.hardwareConcurrency;
		if (!Number.isFinite(navigatorValue)) {
			return resolveFallbackConcurrency(fallback);
		}
		return Math.max(1, Math.floor(navigatorValue as number));
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
