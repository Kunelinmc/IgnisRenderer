import { ShaderMaterial } from "../materials/ShaderMaterial";
import type { Material } from "../materials/Material";
import type { FrameContext } from "./types";
import { ShaderCompileError } from "../shaders/runtime";
import type { ShaderCompilerBackend } from "../shaders/runtime/errorMapping";
import type {
	RenderBackendType,
	WarmupOptions,
	WarmupPhaseReport,
	WarmupReport,
} from "../renderers/IRenderBackend";

export type WarmupSceneTargetMode = "single" | "mrt";

export interface WarmupPlan {
	materials: Material[];
	shaderMaterials: ShaderMaterial[];
	enableSkybox: boolean;
	enableShadows: boolean;
	enableParticles: boolean;
	postProcessPasses: string[];
	sceneTargetMode: WarmupSceneTargetMode;
}

export interface WarmupPhaseCounters extends WarmupPhaseReport {
	errors: ShaderCompileError[];
}

export function buildWarmupPlan(
	context: FrameContext,
	options: WarmupOptions = {}
): WarmupPlan {
	const includeCore = options.includeCorePasses !== false;
	const includeShadow = options.includeShadowPass !== false;
	const includeParticles = options.includeParticles !== false;
	const includePost = options.includePostProcess !== false;

	const materials = collectUniqueMaterials(context);
	const shaderMaterials = materials.filter(
		(material): material is ShaderMaterial => material instanceof ShaderMaterial
	);
	const postProcessPasses =
		includePost ? resolveEnabledPostProcessPasses(context) : [];
	const useMRT = postProcessPasses.length > 0;

	return {
		materials: includeCore ? materials : [],
		shaderMaterials: includeCore ? shaderMaterials : [],
		enableSkybox: includeCore && !!context.scene.skybox && context.features.enableSkybox,
		enableShadows:
			includeShadow &&
			context.features.enableShadows &&
			context.scene.shadowCasterPackets.length > 0,
		enableParticles:
			includeParticles &&
			(context.scene.particleSystems?.length ?? 0) > 0,
		postProcessPasses,
		sceneTargetMode: useMRT ? "mrt" : "single",
	};
}

export function createWarmupReport(backend: RenderBackendType): WarmupReport {
	const now = Date.now();
	return {
		backend,
		startedAt: now,
		finishedAt: now,
		durationMs: 0,
		total: 0,
		compiled: 0,
		skipped: 0,
		failed: 0,
		phases: [],
		errors: [],
	};
}

export function addWarmupPhase(
	report: WarmupReport,
	phase: WarmupPhaseCounters
): void {
	report.total += phase.total;
	report.compiled += phase.compiled;
	report.skipped += phase.skipped;
	report.failed += phase.failed;
	if (phase.errors.length > 0) {
		report.errors.push(...phase.errors);
	}
	report.phases.push({
		phase: phase.phase,
		total: phase.total,
		compiled: phase.compiled,
		skipped: phase.skipped,
		failed: phase.failed,
	});
}

export function finalizeWarmupReport(report: WarmupReport): WarmupReport {
	report.finishedAt = Date.now();
	report.durationMs = Math.max(0, report.finishedAt - report.startedAt);
	return report;
}

export function toShaderCompileError(
	error: unknown,
	backend: RenderBackendType,
	label: string
): ShaderCompileError {
	if (error instanceof ShaderCompileError) {
		return error;
	}
	const compilerBackend: ShaderCompilerBackend =
		(backend === "webgpu" || backend === "webgl" ? backend : "unknown") as
			ShaderCompilerBackend;
	return new ShaderCompileError({
		backend: compilerBackend,
		language: "wgsl",
		stage: "unknown",
		label,
		sourceKind: "unknown",
		code: "",
		sourceMap: null,
		messages: [
			{
				type: "error",
				message: String(error),
			},
		],
		cause: error,
	});
}

function collectUniqueMaterials(context: FrameContext): Material[] {
	const unique = new Set<Material>();
	const packets = [
		...context.scene.opaquePackets,
		...context.scene.transparentPackets,
		...context.scene.shadowCasterPackets,
		...context.scene.reflectivePackets,
	];
	for (const packet of packets) {
		if (packet.material) {
			unique.add(packet.material);
		}
	}
	return Array.from(unique);
}

function resolveEnabledPostProcessPasses(context: FrameContext): string[] {
	const passes: string[] = [];
	if (context.features.enableSSAO) passes.push("ssao");
	if (context.features.enableTAA) passes.push("taa");
	if (context.features.enableSSR) passes.push("ssr");
	if (context.features.enableVolumetric) passes.push("volumetric");
	if (context.features.enableMotionBlur) passes.push("motion-blur");
	if (context.features.enableDOF) passes.push("dof");
	if (context.features.enableBloom) passes.push("bloom");
	if (context.features.enableFXAA) passes.push("fxaa");
	if (context.features.enableGamma) passes.push("gamma");
	return passes;
}
