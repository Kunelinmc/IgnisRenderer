import {
	createRenderBackendExtensionRegistry,
	type RenderBackendExtensionRegistry,
} from "../BackendExtensions";
import {
	assertIBLPrefilterSourceRevision,
	IBL_PREFILTER_EXECUTOR_EXTENSION,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
} from "../../lights/ibl/IBLPrefilterExecutor";

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
	private readonly _iblPrefilterExecutor: IBLPrefilterExecutorLike;
	private readonly _options: WebGLBackendExtensionOwnerOptions;

	public constructor(options: WebGLBackendExtensionOwnerOptions) {
		this._options = options;
		this._iblPrefilterExecutor = {
			id: "webgl",
			getAvailability: () => this._getIBLPrefilterAvailability(),
			execute: (request) => this._executeIBLPrefilter(request),
		};
		this.registry = createRenderBackendExtensionRegistry([
			{
				id: IBL_PREFILTER_EXECUTOR_EXTENSION.id,
				insertionPoints: ["application:ibl-prefilter"],
				api: this._iblPrefilterExecutor,
			},
		]);
	}

	private _getIBLPrefilterAvailability(): IBLPrefilterExecutorAvailability {
		const queueState = this._options.contextWorkQueue
			.getDebugSnapshot().state;
		if (queueState === "destroyed") {
			return {
				state: "unsupported",
				acceptsRequests: false,
				reason: "WebGL IBL prefilter executor has been destroyed.",
			};
		}
		if (queueState === "context-lost") {
			return {
				state: "temporarily-unavailable",
				acceptsRequests: true,
				reason:
					"WebGL IBL prefilter executor is waiting for context restoration.",
			};
		}
		const services = this._options.resolveContextServices();
		if (queueState !== "ready" || !services) {
			return {
				state: "temporarily-unavailable",
				acceptsRequests: false,
				reason:
					"WebGL backend must be initialized before requesting IBL prefiltering.",
			};
		}
		return services.iblPrefilter.getAvailability();
	}

	private _executeIBLPrefilter(
		request: IBLPrefilterExecutionRequest,
	): Promise<IBLPrefilterMipData[]> {
		return this._options.contextWorkQueue.enqueue({
			label: "ibl-prefilter",
			framePolicy: "between-passes",
			contextLossPolicy: "retain-pending",
			signal: request.signal,
			execute: (scope) => {
				assertIBLPrefilterSourceRevision(
					request.envMap,
					request.sourceRevision,
				);
				const availability = scope.services.iblPrefilter.getAvailability();
				if (!availability.acceptsRequests) {
					throw new Error(
						availability.reason ??
							"WebGL IBL prefilter executor is unavailable.",
					);
				}
				return scope.services.iblPrefilter.execute({
					...request,
					signal: scope.signal,
				});
			},
		});
	}
}
