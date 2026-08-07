import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";

import {
	WebGPUFrameConfigurationResolver,
	type WebGPUFrameCapabilitySnapshot,
	type WebGPUFrameConfiguration,
	type WebGPUFrameConfigurationOptions,
} from "./WebGPUFrameConfigurationResolver";
import {
	WebGPUFrameConfigurationBuilder,
	type WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";

/** @internal Aggregates analyzed frame work into one effective configuration. */
export class WebGPUFrameConfigurationModule {
	private readonly _resolver = new WebGPUFrameConfigurationResolver();

	public resolve(
		framePackets: PreparedFramePacketSet,
		contributions: readonly WebGPUFrameModuleConfigurationContribution[],
		capabilities: WebGPUFrameCapabilitySnapshot,
		options: WebGPUFrameConfigurationOptions,
	): WebGPUFrameConfiguration {
		const builder = new WebGPUFrameConfigurationBuilder(framePackets);
		for (const contribution of contributions) contribution(builder);
		const analysis = builder.build();
		return this._resolver.resolve(analysis, capabilities, options);
	}
}
