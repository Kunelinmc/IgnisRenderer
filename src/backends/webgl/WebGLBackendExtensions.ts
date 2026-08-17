import {
	createRenderBackendExtensionRegistry,
	type BackendExtensionAvailability,
	type RenderBackendExtensionRegistry,
} from "../BackendExtensions";
import {
	WEBGL_AUXILIARY_RASTER_EXTENSION,
	type IWebGLAuxiliaryRasterFacade,
	type WebGLAuxiliaryRasterAvailabilityOptions,
	type WebGLAuxiliaryRasterRequest,
	type WebGLAuxiliaryRasterRequirements,
} from "./WebGLAuxiliaryRaster";

import type { WebGLContextServiceOwner } from "./WebGLContextServiceOwner";
import type { WebGLContextWorkQueue } from "./WebGLContextWorkQueue";

interface WebGLBackendExtensionOwnerOptions {
	readonly contextWorkQueue:
		WebGLContextWorkQueue<WebGLContextServiceOwner>;
	resolveContextServices(): WebGLContextServiceOwner | null;
}

/** Owns identity-stable WebGL extension facades. */
export class WebGLBackendExtensionOwner {
	public readonly registry: RenderBackendExtensionRegistry;
	private readonly _auxiliaryRaster: IWebGLAuxiliaryRasterFacade;
	private readonly _options: WebGLBackendExtensionOwnerOptions;

	public constructor(options: WebGLBackendExtensionOwnerOptions) {
		this._options = options;
		this._auxiliaryRaster = {
			getAvailability: (availabilityOptions) =>
				this._getAuxiliaryRasterAvailability(availabilityOptions),
			execute: (request) => this._executeAuxiliaryRaster(request),
		};
		this.registry = createRenderBackendExtensionRegistry([
			{
				id: WEBGL_AUXILIARY_RASTER_EXTENSION.id,
				insertionPoints: ["application:webgl-auxiliary-raster"],
				api: this._auxiliaryRaster,
			},
		]);
	}

	private _getAuxiliaryRasterAvailability(
		options: WebGLAuxiliaryRasterAvailabilityOptions = {},
	): BackendExtensionAvailability {
		const queueState = this._options.contextWorkQueue
			.getDebugSnapshot().state;
		if (queueState === "destroyed") {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason: "WebGL auxiliary raster facade has been destroyed.",
			};
		}
		if (queueState === "context-lost") {
			const retains = options.contextLossPolicy === "retain-pending";
			return {
				state: "temporarily-unavailable",
				acceptsRequests: retains,
				reason: retains ?
					"WebGL auxiliary raster work is waiting for context restoration." :
					"WebGL context is lost and this request rejects on context loss.",
			};
		}
		const services = this._options.resolveContextServices();
		if (queueState !== "ready" || !services) {
			return {
				state: "temporarily-unavailable",
				acceptsRequests: false,
				reason:
					"WebGL backend must be initialized before requesting auxiliary raster work.",
			};
		}
		return resolveRequirementsAvailability(services, options);
	}

	private _executeAuxiliaryRaster<T>(
		request: WebGLAuxiliaryRasterRequest<T>,
	): Promise<T> {
		const framePolicy = request.framePolicy ?? "idle-only";
		const contextLossPolicy = request.contextLossPolicy ?? "reject";
		assertFramePolicy(framePolicy);
		assertContextLossPolicy(contextLossPolicy);
		return this._options.contextWorkQueue.enqueue({
			label: request.label,
			framePolicy,
			contextLossPolicy,
			signal: request.signal,
			execute: (scope) => {
				const availability = resolveRequirementsAvailability(
					scope.services,
					request,
				);
				if (!availability.acceptsRequests) {
					throw new Error(
						availability.reason ??
							"WebGL auxiliary raster requirements are unavailable.",
					);
				}
				return scope.services.auxiliaryRaster.execute(
					scope.generation,
					scope.signal,
					request.task,
				);
			},
		});
	}
}

function resolveRequirementsAvailability(
	services: WebGLContextServiceOwner,
	requirements: WebGLAuxiliaryRasterRequirements,
): BackendExtensionAvailability {
	for (const extension of requirements.requiredExtensions ?? []) {
		if (!services.auxiliaryRaster.hasExtension(extension)) {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason: `WebGL auxiliary raster requires ${extension}.`,
			};
		}
	}
	for (const group of requirements.alternativeExtensionGroups ?? []) {
		if (group.length === 0) {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason: "WebGL auxiliary raster extension alternatives must not be empty.",
			};
		}
		if (!group.some((extension) =>
			services.auxiliaryRaster.hasExtension(extension))) {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason:
					`WebGL auxiliary raster requires one of ${group.join(", ")}.`,
			};
		}
	}
	return { state: "ready", acceptsRequests: true, reason: null };
}

function assertFramePolicy(value: string): asserts value is
	"between-passes" | "idle-only" {
	if (value === "between-passes" || value === "idle-only") return;
	throw new Error(`Unsupported WebGL auxiliary raster frame policy "${value}".`);
}

function assertContextLossPolicy(value: string): asserts value is
	"reject" | "retain-pending" {
	if (value === "reject" || value === "retain-pending") return;
	throw new Error(
		`Unsupported WebGL auxiliary raster context-loss policy "${value}".`,
	);
}
