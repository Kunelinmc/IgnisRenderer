import {
	type BuiltinFramePassStage,
	type FramePassStage,
	type ResolvedFeatureState,
} from "./types";
import {
	getEnabledCustomPostProcessPassIds,
	isFogPostProcessEnabled,
	type PostProcessPassId,
	type ResolvedPostProcessState,
} from "./PostProcessController";

export const RENDER_DIRTY_REASON_MASK = {
	unknown: 1 << 0,
	resize: 1 << 1,
	camera: 1 << 2,
	transform: 1 << 3,
	material: 1 << 4,
	texture: 1 << 5,
	lighting: 1 << 6,
	shadow: 1 << 7,
	postfx: 1 << 8,
	interaction: 1 << 9,
	physics: 1 << 10,
	particles: 1 << 11,
	"postfx-light": 1 << 12,
	"postfx-standard": 1 << 13,
	"postfx-cinematic": 1 << 14,
	"reflection-probe": 1 << 15,
	"environment-ibl": 1 << 16,
	"environment-ibl-complete": 1 << 17,
} as const;

export type BuiltinRenderDirtyReason = keyof typeof RENDER_DIRTY_REASON_MASK;
export type RenderDirtyReason = BuiltinRenderDirtyReason | (string & {});

export const RENDER_DIRTY_GROUP = {
	postfx:
		RENDER_DIRTY_REASON_MASK.postfx |
		RENDER_DIRTY_REASON_MASK["postfx-light"] |
		RENDER_DIRTY_REASON_MASK["postfx-standard"] |
		RENDER_DIRTY_REASON_MASK["postfx-cinematic"],
	shading:
		RENDER_DIRTY_REASON_MASK.material |
		RENDER_DIRTY_REASON_MASK.texture |
		RENDER_DIRTY_REASON_MASK.lighting |
		RENDER_DIRTY_REASON_MASK.shadow |
		RENDER_DIRTY_REASON_MASK["reflection-probe"],
} as const;

export interface DirtyRect {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface DirtyTileCoverage {
	tileSize: number;
	tileColumns: number;
	tileRows: number;
	dirtyTiles: number[];
}

export interface IncrementalRenderingOptions {
	enabled: boolean;
	maxDirtyRects: number;
	dirtyTileSize: number;
	fullFrameFallbackAreaRatio: number;
	temporalPolicy: "conservative-reset";
}

export interface IncrementalFrameStats {
	enabled: boolean;
	reasonMask: number;
	forceFullFrame: boolean;
	temporalHistoryReset: boolean;
	firstPass: FramePassStage | null;
	dirtyRectCount: number;
	dirtyTileCount: number;
	dirtyTileSize: number;
	dirtyTileColumns: number;
	dirtyTileRows: number;
	dirtyAreaRatio: number;
	dirtyRects: DirtyRect[];
	dirtyTiles: number[];
}

export interface IncrementalFrameContext {
	enabled: boolean;
	forceFullFrame: boolean;
	dirtyRects: DirtyRect[];
	dirtyTileSize: number;
	dirtyTileColumns: number;
	dirtyTileRows: number;
	dirtyTiles: number[];
	dirtyAreaRatio: number;
	firstPass: FramePassStage | null;
	reasonMask: number;
	temporalHistoryReset: boolean;
}

export interface IncrementalPlanInput {
	enabled: boolean;
	reasonMask: number;
	features: ResolvedFeatureState;
	postProcess: ResolvedPostProcessState;
	registry?: IncrementalRegistry;
}

export interface IncrementalPlan {
	firstPass: FramePassStage | null;
	forceFullFrame: boolean;
	temporalHistoryReset: boolean;
	reasonMask: number;
}

export const POST_PROCESS_GRADES = [
	"none",
	"light",
	"standard",
	"cinematic",
] as const;
export type PostProcessGrade = (typeof POST_PROCESS_GRADES)[number];

export type IncrementalDirtyReasonGroup =
	| "simulation"
	| "geometry"
	| "postfx"
	| "interaction"
	| (string & {});

export type IncrementalFirstPassResolver = (
	input: IncrementalPlanInput,
	registry: IncrementalRegistry
) => FramePassStage | null;

export interface IncrementalDirtyReasonDescriptor {
	readonly id: string;
	readonly groups?: readonly IncrementalDirtyReasonGroup[];
	readonly firstPass?: FramePassStage | IncrementalFirstPassResolver | null;
	readonly forceFullFrame?: boolean;
	readonly temporalHistoryReset?: boolean;
	readonly invalidatesSceneBounds?: boolean;
}

export interface IncrementalFramePassDescriptor {
	readonly id: FramePassStage;
	readonly order?: number;
}

export interface PostProcessIncrementalMetadata {
	readonly firstPass?: FramePassStage | null;
	readonly grade?: PostProcessGrade;
	readonly inflationRadius?: number;
	readonly fallbackScale?: number;
}

interface RegisteredDirtyReasonDescriptor
	extends IncrementalDirtyReasonDescriptor {
	readonly mask: number;
	readonly groups: readonly IncrementalDirtyReasonGroup[];
	readonly builtIn: boolean;
}

interface RegisteredFramePassDescriptor extends IncrementalFramePassDescriptor {
	readonly order: number;
	readonly builtIn: boolean;
}

interface RegisteredPostProcessIncrementalMetadata
	extends Required<Omit<PostProcessIncrementalMetadata, "firstPass">> {
	readonly id: string;
	readonly firstPass: FramePassStage | null;
	readonly order: number;
	readonly builtIn: boolean;
	readonly isEnabled?: (postProcess: ResolvedPostProcessState) => boolean;
}

interface DirtyReasonSeed extends IncrementalDirtyReasonDescriptor {
	readonly mask: number;
}

interface PostProcessIncrementalSeed
	extends Omit<PostProcessIncrementalMetadata, "firstPass"> {
	readonly id: PostProcessPassId;
	readonly firstPass: PostProcessStage | null;
	readonly order: number;
	readonly isEnabled?: (postProcess: ResolvedPostProcessState) => boolean;
}

const BUILTIN_FRAME_PASS_STAGE_ORDER: FramePassStage[] = [
	"particle-sim",
	"shadow",
	"reflection",
	"main-opaque",
	"main-transparent",
	"particles",
	"ssao",
	"ssgi",
	"taa",
	"ssr",
	"volumetric",
	"fog",
	"motion-blur",
	"dof",
	"bloom",
	"tonemap",
	"color-filter",
	"fxaa",
	"interaction-outline",
	"gamma",
];

type PostProcessStage = Extract<
	BuiltinFramePassStage,
	| "ssao"
	| "ssgi"
	| "taa"
	| "ssr"
	| "volumetric"
	| "fog"
	| "motion-blur"
	| "dof"
	| "bloom"
	| "tonemap"
	| "color-filter"
	| "fxaa"
	| "gamma"
>;

const POST_PROCESS_GRADE_INFLATION_RADIUS: Record<PostProcessGrade, number> = {
	none: 0,
	light: 2,
	standard: 12,
	cinematic: 24,
};

const POST_PROCESS_GRADE_FALLBACK_SCALE: Record<PostProcessGrade, number> = {
	none: 1,
	light: 1,
	standard: 0.9,
	cinematic: 0.8,
};

export const DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE = 32;

export const DEFAULT_INCREMENTAL_RENDERING_OPTIONS: IncrementalRenderingOptions =
	{
		enabled: true,
		maxDirtyRects: 16,
		dirtyTileSize: DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE,
		fullFrameFallbackAreaRatio: 0.3,
		temporalPolicy: "conservative-reset",
	};

const DIRTY_REASON_SEEDS: readonly DirtyReasonSeed[] = [
	{
		id: "unknown",
		mask: RENDER_DIRTY_REASON_MASK.unknown,
		forceFullFrame: true,
		temporalHistoryReset: true,
		invalidatesSceneBounds: true,
	},
	{
		id: "resize",
		mask: RENDER_DIRTY_REASON_MASK.resize,
		groups: ["geometry"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
	{
		id: "camera",
		mask: RENDER_DIRTY_REASON_MASK.camera,
		groups: ["geometry"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
	{
		id: "transform",
		mask: RENDER_DIRTY_REASON_MASK.transform,
		groups: ["geometry"],
		temporalHistoryReset: true,
		invalidatesSceneBounds: true,
	},
	{
		id: "material",
		mask: RENDER_DIRTY_REASON_MASK.material,
		groups: ["geometry"],
		temporalHistoryReset: true,
	},
	{
		id: "texture",
		mask: RENDER_DIRTY_REASON_MASK.texture,
		groups: ["geometry"],
	},
	{
		id: "lighting",
		mask: RENDER_DIRTY_REASON_MASK.lighting,
		groups: ["geometry"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
	{
		id: "shadow",
		mask: RENDER_DIRTY_REASON_MASK.shadow,
		groups: ["geometry"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
	{
		id: "postfx",
		mask: RENDER_DIRTY_REASON_MASK.postfx,
		groups: ["postfx"],
	},
	{
		id: "interaction",
		mask: RENDER_DIRTY_REASON_MASK.interaction,
		groups: ["interaction"],
	},
	{
		id: "physics",
		mask: RENDER_DIRTY_REASON_MASK.physics,
		groups: ["simulation"],
		temporalHistoryReset: true,
		invalidatesSceneBounds: true,
	},
	{
		id: "particles",
		mask: RENDER_DIRTY_REASON_MASK.particles,
		groups: ["simulation"],
		forceFullFrame: true,
		temporalHistoryReset: true,
		invalidatesSceneBounds: true,
	},
	{
		id: "postfx-light",
		mask: RENDER_DIRTY_REASON_MASK["postfx-light"],
		groups: ["postfx"],
	},
	{
		id: "postfx-standard",
		mask: RENDER_DIRTY_REASON_MASK["postfx-standard"],
		groups: ["postfx"],
	},
	{
		id: "postfx-cinematic",
		mask: RENDER_DIRTY_REASON_MASK["postfx-cinematic"],
		groups: ["postfx"],
		temporalHistoryReset: true,
	},
	{
		id: "reflection-probe",
		mask: RENDER_DIRTY_REASON_MASK["reflection-probe"],
		groups: ["geometry"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
	{
		id: "environment-ibl",
		mask: RENDER_DIRTY_REASON_MASK["environment-ibl"],
		forceFullFrame: true,
	},
	{
		id: "environment-ibl-complete",
		mask: RENDER_DIRTY_REASON_MASK["environment-ibl-complete"],
		forceFullFrame: true,
		temporalHistoryReset: true,
	},
];

const POST_PROCESS_INCREMENTAL_SEEDS: readonly PostProcessIncrementalSeed[] = [
	{
		id: "ssao",
		order: 0,
		firstPass: "ssao",
		grade: "standard",
		inflationRadius: 8,
	},
	{
		id: "ssgi",
		order: 1,
		firstPass: "ssgi",
		grade: "standard",
		inflationRadius: 12,
	},
	{
		id: "taa",
		order: 2,
		firstPass: "taa",
		grade: "cinematic",
		inflationRadius: 8,
	},
	{
		id: "ssr",
		order: 3,
		firstPass: "ssr",
		grade: "cinematic",
		inflationRadius: 16,
	},
	{
		id: "volumetric",
		order: 4,
		firstPass: "volumetric",
		grade: "cinematic",
		inflationRadius: 16,
	},
	{
		id: "fog",
		order: 5,
		firstPass: "fog",
		grade: "cinematic",
		inflationRadius: 20,
		isEnabled: isFogPostProcessEnabled,
	},
	{
		id: "motion-blur",
		order: 6,
		firstPass: "motion-blur",
		grade: "cinematic",
		inflationRadius: 24,
	},
	{
		id: "dof",
		order: 7,
		firstPass: "dof",
		grade: "cinematic",
		inflationRadius: 32,
	},
	{
		id: "bloom",
		order: 8,
		firstPass: "bloom",
		grade: "standard",
		inflationRadius: 48,
	},
	{
		id: "tonemap",
		order: 9,
		firstPass: "tonemap",
		grade: "light",
		inflationRadius: 0,
	},
	{
		id: "color-filter",
		order: 10,
		firstPass: "color-filter",
		grade: "light",
		inflationRadius: 2,
	},
	{
		id: "fxaa",
		order: 11,
		firstPass: "fxaa",
		grade: "light",
		inflationRadius: 2,
	},
	{
		id: "gamma",
		order: 12,
		firstPass: "gamma",
		grade: "light",
		inflationRadius: 0,
	},
];

const POST_PROCESS_GRADE_INDEX: Record<PostProcessGrade, number> = {
	none: 0,
	light: 1,
	standard: 2,
	cinematic: 3,
};

const DEFAULT_CUSTOM_POST_PROCESS_INCREMENTAL_METADATA: Required<
	Omit<PostProcessIncrementalMetadata, "firstPass">
> & {
	firstPass: FramePassStage;
} = {
	firstPass: "gamma",
	grade: "light",
	inflationRadius: 2,
	fallbackScale: 1,
};

export class IncrementalRegistry {
	private _dirtyReasons = new Map<string, RegisteredDirtyReasonDescriptor>();
	private _dirtyReasonsByMask = new Map<number, RegisteredDirtyReasonDescriptor>();
	private _framePasses = new Map<FramePassStage, RegisteredFramePassDescriptor>();
	private _postProcessPasses =
		new Map<string, RegisteredPostProcessIncrementalMetadata>();
	private _nextCustomDirtyReasonBit = 18;
	private _nextFramePassOrder = 0;
	private _nextPostProcessOrder = POST_PROCESS_INCREMENTAL_SEEDS.length;

	constructor() {
		for (const descriptor of DIRTY_REASON_SEEDS) {
			this._registerDirtyReason(descriptor, true);
		}
		for (let order = 0; order < BUILTIN_FRAME_PASS_STAGE_ORDER.length; order++) {
			this.registerFramePass(
				{
					id: BUILTIN_FRAME_PASS_STAGE_ORDER[order],
					order,
				},
				true
			);
		}
		for (const metadata of POST_PROCESS_INCREMENTAL_SEEDS) {
			this._registerPostProcessPass(metadata, true);
		}
	}

	/**
	 * Registers a custom dirty reason and allocates a stable bit mask for it.
	 *
	 * @param descriptor Dirty reason behavior used by incremental planning.
	 * @returns The allocated dirty reason mask.
	 * @sideEffects Mutates the registry. Allocated bits are not reused after
	 * unregistering a custom reason.
	 */
	public registerDirtyReason(
		descriptor: IncrementalDirtyReasonDescriptor
	): number {
		return this._registerDirtyReason(descriptor, false).mask;
	}

	/**
	 * Unregisters a custom dirty reason.
	 *
	 * @param id Custom dirty reason id to remove.
	 * @sideEffects Mutates the registry. Built-in reasons cannot be removed.
	 */
	public unregisterDirtyReason(id: string): void {
		const descriptor = this._dirtyReasons.get(id);
		if (!descriptor) {
			return;
		}
		if (descriptor.builtIn) {
			throw new Error(`Cannot unregister built-in dirty reason "${id}".`);
		}
		this._dirtyReasons.delete(id);
		this._dirtyReasonsByMask.delete(descriptor.mask);
	}

	public getDirtyReasonMask(reason: RenderDirtyReason | undefined): number {
		if (!reason) {
			return RENDER_DIRTY_REASON_MASK.unknown;
		}
		return (
			this._dirtyReasons.get(reason)?.mask ??
			RENDER_DIRTY_REASON_MASK.unknown
		);
	}

	public hasAnyDirtyReason(
		mask: number,
		...reasons: RenderDirtyReason[]
	): boolean {
		for (const reason of reasons) {
			if ((mask & this.getDirtyReasonMask(reason)) !== 0) {
				return true;
			}
		}
		return false;
	}

	public doesDirtyReasonInvalidateSceneBounds(
		reason: RenderDirtyReason
	): boolean {
		const descriptor = this._dirtyReasons.get(reason);
		if (!descriptor) {
			return true;
		}
		return descriptor.invalidatesSceneBounds === true;
	}

	public getDirtyReasonDescriptorsForMask(
		mask: number
	): RegisteredDirtyReasonDescriptor[] {
		const normalizedMask = mask >>> 0;
		const descriptors: RegisteredDirtyReasonDescriptor[] = [];
		let matchedMask = 0;
		for (const descriptor of this._dirtyReasonsByMask.values()) {
			if ((normalizedMask & descriptor.mask) === 0) {
				continue;
			}
			descriptors.push(descriptor);
			matchedMask |= descriptor.mask;
		}
		if ((normalizedMask & ~matchedMask) !== 0) {
			const unknown = this._dirtyReasons.get("unknown");
			if (unknown && !descriptors.includes(unknown)) {
				descriptors.push(unknown);
			}
		}
		return descriptors;
	}

	public registerFramePass(
		descriptor: IncrementalFramePassDescriptor,
		builtIn = false
	): void {
		if (!descriptor.id) {
			throw new Error("Incremental frame pass id is required.");
		}
		const current = this._framePasses.get(descriptor.id);
		if (current?.builtIn && !builtIn) {
			throw new Error(
				`Cannot register built-in incremental frame pass "${descriptor.id}".`
			);
		}
		if (
			current?.builtIn &&
			builtIn &&
			descriptor.order === undefined
		) {
			return;
		}
		const order =
			typeof descriptor.order === "number" &&
			Number.isFinite(descriptor.order) ?
				descriptor.order
			:	this._nextFramePassOrder++;
		this._nextFramePassOrder = Math.max(this._nextFramePassOrder, order + 1);
		this._framePasses.set(descriptor.id, {
			id: descriptor.id,
			order,
			builtIn,
		});
	}

	public unregisterFramePass(id: FramePassStage): void {
		const descriptor = this._framePasses.get(id);
		if (!descriptor) {
			return;
		}
		if (descriptor.builtIn) {
			throw new Error(`Cannot unregister built-in frame pass "${id}".`);
		}
		this._framePasses.delete(id);
	}

	public pickEarliestPass(candidates: FramePassStage[]): FramePassStage {
		let earliest = candidates[0];
		let earliestIndex = this.getFramePassOrder(earliest);
		for (let index = 1; index < candidates.length; index++) {
			const candidate = candidates[index];
			const candidateIndex = this.getFramePassOrder(candidate);
			if (candidateIndex < earliestIndex) {
				earliest = candidate;
				earliestIndex = candidateIndex;
			}
		}
		return earliest;
	}

	public getFramePassOrder(stage: FramePassStage): number {
		return this._framePasses.get(stage)?.order ?? Number.MAX_SAFE_INTEGER;
	}

	public registerPostProcessPass(
		id: string,
		metadata: PostProcessIncrementalMetadata = {}
	): void {
		if (this._postProcessPasses.get(id)?.builtIn) {
			throw new Error(
				`Cannot register built-in post-process incremental metadata "${id}".`
			);
		}
		this._registerPostProcessPass({ id, ...metadata }, false);
	}

	public unregisterPostProcessPass(id: string): void {
		const metadata = this._postProcessPasses.get(id);
		if (!metadata) {
			return;
		}
		if (metadata.builtIn) {
			throw new Error(
				`Cannot unregister built-in post-process incremental metadata "${id}".`
			);
		}
		this._postProcessPasses.delete(id);
	}

	public resolveFirstEnabledPostProcessStage(
		postProcess: ResolvedPostProcessState
	): FramePassStage | null {
		const candidates: FramePassStage[] = [];
		for (const metadata of this.getEnabledPostProcessMetadata(postProcess)) {
			if (metadata.firstPass) {
				candidates.push(metadata.firstPass);
			}
		}
		return candidates.length > 0 ? this.pickEarliestPass(candidates) : null;
	}

	public resolvePostProcessGrade(
		postProcess: ResolvedPostProcessState
	): PostProcessGrade {
		let grade: PostProcessGrade = "none";
		for (const metadata of this.getEnabledPostProcessMetadata(postProcess)) {
			if (
				POST_PROCESS_GRADE_INDEX[metadata.grade] >
				POST_PROCESS_GRADE_INDEX[grade]
			) {
				grade = metadata.grade;
			}
		}
		return grade;
	}

	public computePostProcessInflationRadius(
		postProcess: ResolvedPostProcessState
	): number {
		let radius = getPostProcessGradeInflationRadius(
			this.resolvePostProcessGrade(postProcess)
		);
		for (const metadata of this.getEnabledPostProcessMetadata(postProcess)) {
			radius = Math.max(radius, metadata.inflationRadius);
		}
		return radius;
	}

	public scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio: number,
		postProcess: ResolvedPostProcessState
	): number {
		const normalizedBaseRatio = clampNumber(
			baseRatio,
			0.01,
			1,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.fullFrameFallbackAreaRatio
		);
		let scale =
			POST_PROCESS_GRADE_FALLBACK_SCALE[
				this.resolvePostProcessGrade(postProcess)
			] ?? 1;
		for (const metadata of this.getEnabledPostProcessMetadata(postProcess)) {
			scale = Math.min(scale, metadata.fallbackScale);
		}
		return clampNumber(normalizedBaseRatio * scale, 0.01, 1, normalizedBaseRatio);
	}

	private getEnabledPostProcessMetadata(
		postProcess: ResolvedPostProcessState
	): RegisteredPostProcessIncrementalMetadata[] {
		const metadataList: RegisteredPostProcessIncrementalMetadata[] = [];
		const seen = new Set<string>();
		for (const metadata of this._postProcessPasses.values()) {
			if (!this._isPostProcessMetadataEnabled(metadata, postProcess)) {
				continue;
			}
			metadataList.push(metadata);
			seen.add(metadata.id);
		}
		for (const id of getEnabledCustomPostProcessPassIds(postProcess)) {
			if (seen.has(id)) {
				continue;
			}
			metadataList.push(this._createDefaultCustomPostProcessMetadata(id));
		}
		metadataList.sort((left, right) => left.order - right.order);
		return metadataList;
	}

	private _isPostProcessMetadataEnabled(
		metadata: RegisteredPostProcessIncrementalMetadata,
		postProcess: ResolvedPostProcessState
	): boolean {
		if (metadata.isEnabled) {
			return metadata.isEnabled(postProcess);
		}
		return postProcess.enabled[metadata.id] === true;
	}

	private _registerDirtyReason(
		descriptor: IncrementalDirtyReasonDescriptor & { readonly mask?: number },
		builtIn: boolean
	): RegisteredDirtyReasonDescriptor {
		if (!descriptor.id) {
			throw new Error("Dirty reason id is required.");
		}
		if (this._dirtyReasons.has(descriptor.id)) {
			throw new Error(`Dirty reason "${descriptor.id}" is already registered.`);
		}
		const mask = builtIn ? descriptor.mask : this._allocateCustomDirtyReasonMask();
		if (typeof mask !== "number" || !Number.isFinite(mask) || mask <= 0) {
			throw new Error(`Dirty reason "${descriptor.id}" requires a valid mask.`);
		}
		const normalizedMask = mask >>> 0;
		if (this._dirtyReasonsByMask.has(normalizedMask)) {
			throw new Error(
				`Dirty reason mask ${normalizedMask} is already registered.`
			);
		}
		const registered: RegisteredDirtyReasonDescriptor = {
			id: descriptor.id,
			mask: normalizedMask,
			groups: descriptor.groups ? descriptor.groups.slice() : [],
			firstPass: descriptor.firstPass,
			forceFullFrame: descriptor.forceFullFrame === true,
			temporalHistoryReset: descriptor.temporalHistoryReset === true,
			invalidatesSceneBounds: descriptor.invalidatesSceneBounds === true,
			builtIn,
		};
		this._dirtyReasons.set(registered.id, registered);
		this._dirtyReasonsByMask.set(registered.mask, registered);
		return registered;
	}

	private _allocateCustomDirtyReasonMask(): number {
		if (this._nextCustomDirtyReasonBit >= 31) {
			throw new Error("No incremental dirty reason bits are available.");
		}
		const mask = 1 << this._nextCustomDirtyReasonBit;
		this._nextCustomDirtyReasonBit++;
		return mask >>> 0;
	}

	private _registerPostProcessPass(
		metadata: PostProcessIncrementalMetadata & {
			readonly id: string;
			readonly order?: number;
			readonly isEnabled?: (postProcess: ResolvedPostProcessState) => boolean;
		},
		builtIn: boolean
	): void {
		if (!metadata.id) {
			throw new Error("Post-process incremental metadata id is required.");
		}
		const current = this._postProcessPasses.get(metadata.id);
		if (current?.builtIn && !builtIn) {
			throw new Error(
				`Cannot register built-in post-process incremental metadata "${metadata.id}".`
			);
		}
		const order =
			typeof metadata.order === "number" && Number.isFinite(metadata.order) ?
				metadata.order
			:	this._nextPostProcessOrder++;
		this._nextPostProcessOrder = Math.max(
			this._nextPostProcessOrder,
			order + 1
		);
		const normalized =
			builtIn ?
				normalizeBuiltInPostProcessMetadata(metadata, order)
			:	this._createDefaultCustomPostProcessMetadata(metadata.id, metadata, order);
		this._postProcessPasses.set(metadata.id, {
			...normalized,
			order,
			builtIn,
			isEnabled: metadata.isEnabled,
		});
	}

	private _createDefaultCustomPostProcessMetadata(
		id: string,
		metadata: PostProcessIncrementalMetadata = {},
		order = this._nextPostProcessOrder
	): RegisteredPostProcessIncrementalMetadata {
		return {
			id,
			firstPass:
				metadata.firstPass === undefined ?
					DEFAULT_CUSTOM_POST_PROCESS_INCREMENTAL_METADATA.firstPass
				:	metadata.firstPass,
			grade:
				metadata.grade ??
				DEFAULT_CUSTOM_POST_PROCESS_INCREMENTAL_METADATA.grade,
			inflationRadius:
				metadata.inflationRadius ??
				DEFAULT_CUSTOM_POST_PROCESS_INCREMENTAL_METADATA.inflationRadius,
			fallbackScale:
				metadata.fallbackScale ??
				DEFAULT_CUSTOM_POST_PROCESS_INCREMENTAL_METADATA.fallbackScale,
			order,
			builtIn: false,
		};
	}
}

const DEFAULT_INCREMENTAL_REGISTRY = new IncrementalRegistry();

export function getDefaultIncrementalRegistry(): IncrementalRegistry {
	return DEFAULT_INCREMENTAL_REGISTRY;
}

export function registerRenderDirtyReason(
	descriptor: IncrementalDirtyReasonDescriptor
): number {
	return DEFAULT_INCREMENTAL_REGISTRY.registerDirtyReason(descriptor);
}

export function unregisterRenderDirtyReason(id: string): void {
	DEFAULT_INCREMENTAL_REGISTRY.unregisterDirtyReason(id);
}

export function doesRenderDirtyReasonInvalidateSceneBounds(
	reason: RenderDirtyReason
): boolean {
	return DEFAULT_INCREMENTAL_REGISTRY.doesDirtyReasonInvalidateSceneBounds(
		reason
	);
}

export function renderDirtyReasonToMask(
	reason: RenderDirtyReason | undefined
): number {
	return DEFAULT_INCREMENTAL_REGISTRY.getDirtyReasonMask(reason);
}

export function hasAnyDirtyReason(
	mask: number,
	...reasons: RenderDirtyReason[]
): boolean {
	return DEFAULT_INCREMENTAL_REGISTRY.hasAnyDirtyReason(mask, ...reasons);
}

export function normalizeIncrementalRenderingOptions(
	options?: Partial<IncrementalRenderingOptions> | null
): IncrementalRenderingOptions {
	const source = options ?? {};
	return {
		enabled: source.enabled ?? DEFAULT_INCREMENTAL_RENDERING_OPTIONS.enabled,
		maxDirtyRects: clampInteger(
			source.maxDirtyRects,
			1,
			256,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.maxDirtyRects
		),
		dirtyTileSize: clampInteger(
			source.dirtyTileSize,
			4,
			512,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.dirtyTileSize
		),
		fullFrameFallbackAreaRatio: clampNumber(
			source.fullFrameFallbackAreaRatio,
			0.01,
			1,
			DEFAULT_INCREMENTAL_RENDERING_OPTIONS.fullFrameFallbackAreaRatio
		),
		temporalPolicy: "conservative-reset",
	};
}

export function mergeIncrementalRenderingOptions(
	current: IncrementalRenderingOptions,
	next?: Partial<IncrementalRenderingOptions> | null
): IncrementalRenderingOptions {
	if (!next) {
		return current;
	}
	return normalizeIncrementalRenderingOptions({
		...current,
		...next,
	});
}

export class IncrementalFramePlanner {
	public static plan(input: IncrementalPlanInput): IncrementalPlan {
		const reasonMask = input.reasonMask >>> 0;
		const registry = input.registry ?? DEFAULT_INCREMENTAL_REGISTRY;
		if (!input.enabled) {
			return {
				firstPass: null,
				forceFullFrame: true,
				temporalHistoryReset: true,
				reasonMask,
			};
		}

		if (reasonMask === 0) {
			return {
				firstPass: null,
				forceFullFrame: false,
				temporalHistoryReset: false,
				reasonMask,
			};
		}

		const candidates: FramePassStage[] = [];
		let temporalHistoryReset = false;
		let forceFullFrame = false;
		for (const descriptor of registry.getDirtyReasonDescriptorsForMask(reasonMask)) {
			forceFullFrame ||= descriptor.forceFullFrame === true;
			temporalHistoryReset ||= descriptor.temporalHistoryReset === true;
			const firstPass = resolveDirtyReasonFirstPass(
				descriptor,
				input,
				registry
			);
			if (firstPass) {
				candidates.push(firstPass);
			}
		}

		if (candidates.length === 0) {
			candidates.push("main-opaque");
		}

		return {
			firstPass: registry.pickEarliestPass(candidates),
			forceFullFrame,
			temporalHistoryReset,
			reasonMask,
		};
	}
}

export function makeFullScreenRect(width: number, height: number): DirtyRect {
	return {
		x: 0,
		y: 0,
		width: Math.max(1, Math.floor(width)),
		height: Math.max(1, Math.floor(height)),
	};
}

export function clampDirtyRect(
	rect: DirtyRect,
	width: number,
	height: number
): DirtyRect | null {
	const maxWidth = Math.max(1, Math.floor(width));
	const maxHeight = Math.max(1, Math.floor(height));
	const minX = Math.max(0, Math.floor(rect.x));
	const minY = Math.max(0, Math.floor(rect.y));
	const maxX = Math.min(maxWidth, Math.ceil(rect.x + rect.width));
	const maxY = Math.min(maxHeight, Math.ceil(rect.y + rect.height));
	const clampedWidth = maxX - minX;
	const clampedHeight = maxY - minY;
	if (clampedWidth <= 0 || clampedHeight <= 0) {
		return null;
	}
	return {
		x: minX,
		y: minY,
		width: clampedWidth,
		height: clampedHeight,
	};
}

export function buildDirtyTileCoverage(
	rects: DirtyRect[],
	width: number,
	height: number,
	tileSize: number
): DirtyTileCoverage {
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const resolvedTileSize = clampInteger(
		tileSize,
		4,
		512,
		DEFAULT_INCREMENTAL_DIRTY_TILE_SIZE
	);
	const tileColumns = Math.max(1, Math.ceil(resolvedWidth / resolvedTileSize));
	const tileRows = Math.max(1, Math.ceil(resolvedHeight / resolvedTileSize));
	const tileCount = tileColumns * tileRows;
	const visited = new Uint8Array(tileCount);
	const dirtyTiles: number[] = [];

	for (const rect of rects) {
		const clamped = clampDirtyRect(rect, resolvedWidth, resolvedHeight);
		if (!clamped) {
			continue;
		}
		const minTileX = Math.floor(clamped.x / resolvedTileSize);
		const minTileY = Math.floor(clamped.y / resolvedTileSize);
		const maxTileX = Math.floor(
			(clamped.x + clamped.width - 1) / resolvedTileSize
		);
		const maxTileY = Math.floor(
			(clamped.y + clamped.height - 1) / resolvedTileSize
		);
		for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
			for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
				const tileIndex = tileY * tileColumns + tileX;
				if (tileIndex < 0 || tileIndex >= tileCount || visited[tileIndex] !== 0) {
					continue;
				}
				visited[tileIndex] = 1;
				dirtyTiles.push(tileIndex);
			}
		}
	}

	dirtyTiles.sort((left, right) => left - right);
	return {
		tileSize: resolvedTileSize,
		tileColumns,
		tileRows,
		dirtyTiles,
	};
}

export function tileCoverageToDirtyRects(
	coverage: DirtyTileCoverage,
	maxRects: number,
	width: number,
	height: number
): DirtyRect[] {
	if (coverage.dirtyTiles.length === 0) {
		return [];
	}
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const tileRects: DirtyRect[] = [];
	for (const tileIndex of coverage.dirtyTiles) {
		const tileX = tileIndex % coverage.tileColumns;
		const tileY = Math.floor(tileIndex / coverage.tileColumns);
		if (
			tileX < 0 ||
			tileY < 0 ||
			tileX >= coverage.tileColumns ||
			tileY >= coverage.tileRows
		) {
			continue;
		}
		const x = tileX * coverage.tileSize;
		const y = tileY * coverage.tileSize;
		tileRects.push({
			x,
			y,
			width: Math.min(coverage.tileSize, resolvedWidth - x),
			height: Math.min(coverage.tileSize, resolvedHeight - y),
		});
	}
	return mergeDirtyRects(tileRects, maxRects, resolvedWidth, resolvedHeight);
}

export function getDirtyTileCoverageAreaRatio(
	coverage: DirtyTileCoverage,
	width: number,
	height: number
): number {
	const resolvedWidth = Math.max(1, Math.floor(width));
	const resolvedHeight = Math.max(1, Math.floor(height));
	const area = resolvedWidth * resolvedHeight;
	let dirtyArea = 0;
	for (const tileIndex of coverage.dirtyTiles) {
		const tileX = tileIndex % coverage.tileColumns;
		const tileY = Math.floor(tileIndex / coverage.tileColumns);
		if (
			tileX < 0 ||
			tileY < 0 ||
			tileX >= coverage.tileColumns ||
			tileY >= coverage.tileRows
		) {
			continue;
		}
		const x = tileX * coverage.tileSize;
		const y = tileY * coverage.tileSize;
		const tileWidth = Math.min(coverage.tileSize, resolvedWidth - x);
		const tileHeight = Math.min(coverage.tileSize, resolvedHeight - y);
		if (tileWidth > 0 && tileHeight > 0) {
			dirtyArea += tileWidth * tileHeight;
		}
	}
	return Math.max(0, Math.min(1, dirtyArea / area));
}

export function inflateDirtyRects(
	rects: DirtyRect[],
	amount: number,
	width: number,
	height: number
): DirtyRect[] {
	const inflateAmount = Math.max(0, Math.floor(amount));
	if (inflateAmount <= 0 || rects.length === 0) {
		return rects.slice();
	}
	const result: DirtyRect[] = [];
	for (const rect of rects) {
		const inflated = clampDirtyRect(
			{
				x: rect.x - inflateAmount,
				y: rect.y - inflateAmount,
				width: rect.width + inflateAmount * 2,
				height: rect.height + inflateAmount * 2,
			},
			width,
			height
		);
		if (inflated) {
			result.push(inflated);
		}
	}
	return result;
}

export function mergeDirtyRects(
	rects: DirtyRect[],
	maxRects: number,
	width: number,
	height: number
): DirtyRect[] {
	const normalized: DirtyRect[] = [];
	for (const rect of rects) {
		const clamped = clampDirtyRect(rect, width, height);
		if (!clamped) continue;
		normalized.push(clamped);
	}
	if (normalized.length <= 1) {
		return normalized;
	}

	normalized.sort((left, right) => {
		if (left.x !== right.x) return left.x - right.x;
		return left.y - right.y;
	});

	const merged: DirtyRect[] = [];
	for (const rect of normalized) {
		let current = rect;
		for (let i = merged.length - 1; i >= 0; i--) {
			const previous = merged[i];
			if (!dirtyRectsIntersectOrTouch(previous, current)) {
				continue;
			}
			current = unionDirtyRect(previous, current);
			merged.splice(i, 1);
		}
		merged.push(current);
	}

	const cappedMaxRects = clampInteger(maxRects, 1, 256, 16);
	while (merged.length > cappedMaxRects) {
		let bestLeft = 0;
		let bestRight = 1;
		let bestGrowth = Number.POSITIVE_INFINITY;
		for (let leftIndex = 0; leftIndex < merged.length; leftIndex++) {
			for (
				let rightIndex = leftIndex + 1;
				rightIndex < merged.length;
				rightIndex++
			) {
				const left = merged[leftIndex];
				const right = merged[rightIndex];
				const union = unionDirtyRect(left, right);
				const growth = getDirtyRectArea(union) -
					(getDirtyRectArea(left) + getDirtyRectArea(right));
				if (growth < bestGrowth) {
					bestGrowth = growth;
					bestLeft = leftIndex;
					bestRight = rightIndex;
				}
			}
		}
		const union = unionDirtyRect(merged[bestLeft], merged[bestRight]);
		merged.splice(bestRight, 1);
		merged.splice(bestLeft, 1, union);
	}

	return merged;
}

export function getDirtyRectsAreaRatio(
	rects: DirtyRect[],
	width: number,
	height: number
): number {
	const area = Math.max(1, Math.floor(width) * Math.floor(height));
	let dirtyArea = 0;
	for (const rect of rects) {
		dirtyArea += getDirtyRectArea(rect);
	}
	return Math.max(0, Math.min(1, dirtyArea / area));
}

export function computePostProcessInflationRadius(
	postProcess: ResolvedPostProcessState
): number {
	return DEFAULT_INCREMENTAL_REGISTRY.computePostProcessInflationRadius(
		postProcess
	);
}

export function resolvePostProcessGrade(
	postProcess: ResolvedPostProcessState
): PostProcessGrade {
	return DEFAULT_INCREMENTAL_REGISTRY.resolvePostProcessGrade(postProcess);
}

export function getPostProcessGradeInflationRadius(
	grade: PostProcessGrade
): number {
	return POST_PROCESS_GRADE_INFLATION_RADIUS[grade] ?? 0;
}

export function scaleFullFrameFallbackAreaRatioForPostProcess(
	baseRatio: number,
	postProcess: ResolvedPostProcessState
): number {
	return DEFAULT_INCREMENTAL_REGISTRY.scaleFullFrameFallbackAreaRatioForPostProcess(
		baseRatio,
		postProcess
	);
}

export function unionDirtyRect(left: DirtyRect, right: DirtyRect): DirtyRect {
	const minX = Math.min(left.x, right.x);
	const minY = Math.min(left.y, right.y);
	const maxX = Math.max(left.x + left.width, right.x + right.width);
	const maxY = Math.max(left.y + left.height, right.y + right.height);
	return {
		x: minX,
		y: minY,
		width: maxX - minX,
		height: maxY - minY,
	};
}

function resolveDirtyReasonFirstPass(
	descriptor: RegisteredDirtyReasonDescriptor,
	input: IncrementalPlanInput,
	registry: IncrementalRegistry
): FramePassStage | null {
	if (typeof descriptor.firstPass === "function") {
		return descriptor.firstPass(input, registry);
	}
	if (descriptor.firstPass) {
		return descriptor.firstPass;
	}
	for (const group of descriptor.groups) {
		switch (group) {
			case "simulation":
				return "particle-sim";
			case "geometry":
				return input.features.enableShadows ? "shadow" : "main-opaque";
			case "postfx":
				return (
					registry.resolveFirstEnabledPostProcessStage(input.postProcess) ??
					"gamma"
				);
			case "interaction":
				return "interaction-outline";
			default:
				break;
		}
	}
	return null;
}

function normalizeBuiltInPostProcessMetadata(
	metadata: PostProcessIncrementalMetadata & {
		readonly id: string;
		readonly order?: number;
		readonly isEnabled?: (postProcess: ResolvedPostProcessState) => boolean;
	},
	order: number
): RegisteredPostProcessIncrementalMetadata {
	return {
		id: metadata.id,
		firstPass: metadata.firstPass ?? null,
		grade: metadata.grade ?? "none",
		inflationRadius: metadata.inflationRadius ?? 0,
		fallbackScale: metadata.fallbackScale ?? 1,
		order,
		builtIn: true,
		isEnabled: metadata.isEnabled,
	};
}

function getDirtyRectArea(rect: DirtyRect): number {
	return Math.max(0, rect.width) * Math.max(0, rect.height);
}

function dirtyRectsIntersectOrTouch(left: DirtyRect, right: DirtyRect): boolean {
	return (
		left.x <= right.x + right.width &&
		left.x + left.width >= right.x &&
		left.y <= right.y + right.height &&
		left.y + left.height >= right.y
	);
}

function clampNumber(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, value));
}

function clampInteger(
	value: number | undefined,
	min: number,
	max: number,
	fallback: number
): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(max, Math.max(min, Math.floor(value)));
}
