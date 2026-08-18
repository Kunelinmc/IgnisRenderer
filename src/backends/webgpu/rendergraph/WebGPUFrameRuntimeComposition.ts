import type { FramePacketProvider } from "../../../pipeline/FramePacketContributorRegistry";
import type { WebGPUFrameServiceOwner } from "../WebGPUFrameServiceOwner";
import { WebGPUHiZBuilder } from "../WebGPUHiZBuilder";
import { WebGPUOcclusionCullingRuntime } from "../WebGPUOcclusionCullingRuntime";
import { WebGPUPlanarReflectionPass } from "../WebGPUPlanarReflectionPass";
import { WebGPUPostProcessRuntime } from "../WebGPUPostProcessRuntime";
import type { WebGPUSampleCountResolver } from "../WebGPUSampleCountResolver";
import { WebGPUCustomRenderTargetRuntime } from "./WebGPUCustomRenderTargetRuntime";
import { WebGPUFrameConfigurationModule } from "./WebGPUFrameConfigurationModule";
import { WebGPUDeferredDecalPass } from "./WebGPUDeferredDecalPass";
import {
	WebGPUDeferredFrameModule,
	WebGPUDeferredOpaqueStatePort,
} from "./WebGPUDeferredFrameModule";
import { WebGPUDeferredLightingPass } from "./WebGPUDeferredLightingPass";
import { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import { WebGPUColorDirtyClearPass } from "./WebGPUColorDirtyClearPass";
import { WebGPUFrameGraphModuleRegistry } from "./WebGPUFrameGraphModuleRegistry";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import { WebGPUPostProcessBridge } from "./WebGPUPostProcessBridge";
import { WebGPUPostProcessFrameModule } from "./WebGPUPostProcessFrameModule";
import { WebGPUPresentationRuntime } from "./WebGPUPresentationRuntime";
import { WebGPUReflectionFrameModule } from "./WebGPUReflectionFrameModule";
import { WebGPUSceneFrameModule } from "./WebGPUSceneFrameModule";
import { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";
import { WebGPUShadowFrameModule } from "./WebGPUShadowFrameModule";
import { WebGPUTransparencyRuntime } from "./WebGPUTransparencyRuntime";
import { WebGPUVisibilityFrameModule } from "./WebGPUVisibilityFrameModule";

/** @internal Sealed backend-private frame runtime composition. */
export interface WebGPUFrameRuntimeComposition {
	readonly modules: WebGPUFrameGraphModuleRegistry;
	readonly lifecycle: Pick<
		WebGPUFrameGraphModuleRegistry,
		"invalidateFrameResources" | "onDisplayOutputChanged" |
		"onShaderRuntimeChanged" | "destroy"
	>;
	readonly postProcess: Pick<
		WebGPUPostProcessFrameModule,
		"getDebugState" | "warmup" | "invalidateFrameResources" |
		"createSessionPort" | "createGBufferBridge" |
		"createPassExecutionContext" | "completePass"
	>;
	readonly visibility: Pick<
		WebGPUVisibilityFrameModule,
		"reset" | "getVisibilityProvider"
	>;
	readonly customRenderTargets: Pick<
		WebGPUCustomRenderTargetRuntime,
		"readColor"
	>;
}

/** Creates one backend-owned, sealed frame runtime composition. */
export function createWebGPUFrameRuntimeComposition(options: {
	readonly host: WebGPUFrameHost;
	readonly frameServices: WebGPUFrameServiceOwner;
	readonly framePackets: FramePacketProvider;
	readonly sampleCountResolver: WebGPUSampleCountResolver;
	warnOnce(code: string, message: string, cause?: unknown): void;
}): WebGPUFrameRuntimeComposition {
	const { host, frameServices, framePackets } = options;
	const particleRenderer = frameServices.getParticleBillboardRenderer();
	const hiZBuilder = new WebGPUHiZBuilder(host.computeFacade);
	const postRuntime = new WebGPUPostProcessRuntime(
		host.computeFacade,
		(key, message) => options.warnOnce(key, message),
		frameServices.sceneFrameLayout,
		hiZBuilder,
		() => host.displayOutputState,
	);
	const planarReflectionPass = new WebGPUPlanarReflectionPass(
		host,
		frameServices,
		framePackets,
	);
	const customRenderTargets = new WebGPUCustomRenderTargetRuntime(
		host,
		options.sampleCountResolver,
	);
	const modules = new WebGPUFrameGraphModuleRegistry();
	const presentation = new WebGPUPresentationRuntime(host, {
		getOutputColorDomain: () => modules.finalOutput.colorDomain,
	});
	const visibility = new WebGPUVisibilityFrameModule(
		hiZBuilder,
		new WebGPUOcclusionCullingRuntime(host),
	);
	const postBridge = new WebGPUPostProcessBridge(host, postRuntime, {
		isHiZReady: () => visibility.isHiZReady(),
	});
	const postProcess = new WebGPUPostProcessFrameModule(
		postRuntime,
		host.postProcessRuntime,
		postBridge,
		presentation,
	);
	const depthDirtyClearPass = new WebGPUDepthDirtyClearPass(host);
	const colorDirtyClearPass = new WebGPUColorDirtyClearPass(host);
	const deferredOpaqueState = new WebGPUDeferredOpaqueStatePort();
	const deferredLightingPass = new WebGPUDeferredLightingPass(host, frameServices);
	const deferredDecalPass = new WebGPUDeferredDecalPass(host, frameServices);
	const sceneRecorder = new WebGPUScenePassRecorder(
		host,
		frameServices,
		particleRenderer,
		depthDirtyClearPass,
		colorDirtyClearPass,
		{
			getGBufferWriteBinding: () => deferredLightingPass.getGBufferWriteBinding(),
			preflightDeferredFrame: async (context) => {
				await deferredLightingPass.preflight();
				await deferredDecalPass.preflight(context);
			},
		},
	);
	const reflection = new WebGPUReflectionFrameModule(
		host,
		planarReflectionPass,
	);
	const configuration = new WebGPUFrameConfigurationModule();
	for (const module of [
		new WebGPUSceneFrameModule(
			sceneRecorder,
			depthDirtyClearPass,
			colorDirtyClearPass,
			deferredOpaqueState,
		),
		new WebGPUShadowFrameModule(frameServices),
		new WebGPUDeferredFrameModule(
			deferredLightingPass,
			deferredDecalPass,
			sceneRecorder,
			deferredOpaqueState,
		),
		new WebGPUTransparencyRuntime(
			host,
			frameServices,
			particleRenderer,
			sceneRecorder,
			{ warnOnce: options.warnOnce },
		),
		reflection,
		visibility,
		postProcess,
		presentation,
		customRenderTargets,
	]) {
		modules.register(module);
	}
	for (const handler of configuration.messageHandlers) {
		modules.registerMessageHandler(handler);
	}
	modules.seal();
	return {
		modules,
		lifecycle: modules,
		postProcess,
		visibility,
		customRenderTargets,
	};
}
