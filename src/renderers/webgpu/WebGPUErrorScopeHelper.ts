import { Logger } from "../../foundation/Logger";

type ScopeLogLevel = "error" | "warn";
const WEBGPU_ERROR_SCOPE_LOGGER = new Logger({
	name: "WebGPUErrorScope",
	level: "warn",
});

interface ScopeOptions {
	level?: ScopeLogLevel;
}

export class WebGPUErrorScopeHelper {
	private _device: GPUDevice;

	constructor(device: GPUDevice) {
		this._device = device;
	}

	public run<T>(
		filter: GPUErrorFilter,
		label: string,
		operation: () => T,
		options?: ScopeOptions
	): T {
		this._device.pushErrorScope(filter);
		try {
			return operation();
		} finally {
			void this._device
				.popErrorScope()
				.then((error) => this._logScopeResult(filter, label, error, options))
				.catch((error) => {
					WEBGPU_ERROR_SCOPE_LOGGER.error(
						`WebGPU ErrorScope pop failed [${label}]: ${String(error)}`
					);
				});
		}
	}

	public async runAsync<T>(
		filter: GPUErrorFilter,
		label: string,
		operation: () => Promise<T>,
		options?: ScopeOptions
	): Promise<T> {
		this._device.pushErrorScope(filter);
		try {
			return await operation();
		} finally {
			try {
				const error = await this._device.popErrorScope();
				this._logScopeResult(filter, label, error, options);
			} catch (popError) {
				WEBGPU_ERROR_SCOPE_LOGGER.error(
					`WebGPU ErrorScope pop failed [${label}]: ${String(popError)}`
				);
			}
		}
	}

	private _logScopeResult(
		filter: GPUErrorFilter,
		label: string,
		error: GPUError | null,
		options?: ScopeOptions
	): void {
		if (!error) {
			return;
		}

		const logLevel = options?.level ?? "error";
		const message = `WebGPU ${filter} error [${label}]: ${error.message}`;
		if (logLevel === "warn") {
			WEBGPU_ERROR_SCOPE_LOGGER.warn(message);
			return;
		}
		WEBGPU_ERROR_SCOPE_LOGGER.error(message);
	}
}
