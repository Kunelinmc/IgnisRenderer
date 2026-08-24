import type {
	FrameContext,
	FramePass,
} from "../../../pipeline/types";
import { materialUsesTransmission } from "../../../materials/transparency";
import type {
	WebGLFrameGraphNode,
	WebGLFrameGraphPlannerState,
	WebGLFrameGraphResourceRef,
	WebGLFrameGraphStagePlan,
} from "./types";

/**
 * Builds WebGL-internal frame graph nodes for renderer-level backend passes.
 */
export class WebGLFrameGraphPlanner {
	private readonly _stagePlanners: Map<
		FramePass["stage"],
		(
			pass: FramePass,
			context: FrameContext,
			state: WebGLFrameGraphPlannerState
		) => WebGLFrameGraphNode[]
	>;

	public constructor() {
		this._stagePlanners = this._createStagePlanners();
	}

	/**
	 * Plans one renderer-level WebGL backend pass.
	 *
	 * @internal Owned by `WebGLFrameGraphRuntime`.
	 * @param pass Renderer-level backend pass.
	 * @param context Active frame context.
	 * @param state WebGL frame graph state for feature-dependent nodes.
	 * @returns Stage plan preserving WebGL execution order.
	 * @sideEffects None.
	 */
	public planStage(
		pass: FramePass,
		context: FrameContext,
		state: WebGLFrameGraphPlannerState
	): WebGLFrameGraphStagePlan {
		const planner = this._stagePlanners.get(pass.stage);
		return {
			pass,
			nodes: planner ? planner(pass, context, state) : [],
		};
	}

	/**
	 * Creates synthetic begin-frame nodes that run before renderer-level passes.
	 *
	 * @internal Used by WebGL frame graph runtime.
	 * @param context Active frame context.
	 * @param state Current WebGL graph state.
	 * @returns Nodes for clear and optional environment background work.
	 * @sideEffects None.
	 */
	public planBeginFrame(
		context: FrameContext,
		state: WebGLFrameGraphPlannerState
	): WebGLFrameGraphNode[] {
		const pass = {
			stage: "webgl-begin-frame",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		} satisfies FramePass;
		const nodes = [
			this._node(pass, "scene-clear", "WebGLSceneClear", {
				writes: [
					this._write("frame:scene-color", "framebuffer-color"),
					this._write("frame:motion-depth", "framebuffer-color"),
					this._write("frame:normal", "framebuffer-color", true),
					this._write("frame:depth", "framebuffer-depth"),
				],
			}),
		];
		if (state.hasEnvironmentBackground) {
			nodes.push(
				this._node(pass, "environment", "WebGLEnvironmentBackground", {
					reads: [
						this._read("environment:background", "texture-sampling", true),
					],
					writes: [
						this._write("frame:scene-color", "framebuffer-color"),
						this._write("frame:motion-depth", "framebuffer-color"),
					],
				})
			);
		}
		void context;
		return nodes;
	}

	/**
	 * Creates a synthetic end-frame present node.
	 *
	 * @internal Used by WebGL frame graph runtime.
	 * @returns Present node.
	 * @sideEffects None.
	 */
	public planPresent(sourceResource = "frame:scene-color"): WebGLFrameGraphNode[] {
		const pass = {
			stage: "webgl-present",
			executor: "backend",
			enabled: true,
			dependsOn: [],
		} satisfies FramePass;
		return [
			this._node(pass, "present", "WebGLPresent", {
				reads: [this._read(sourceResource, "texture-sampling")],
				writes: [this._write("canvas:color", "present")],
			}),
		];
	}

	private _createStagePlanners(): Map<
		FramePass["stage"],
		(
			pass: FramePass,
			context: FrameContext,
			state: WebGLFrameGraphPlannerState
		) => WebGLFrameGraphNode[]
	> {
		return new Map([
			[
				"shadow",
				(pass) => [
					this._node(pass, "shadow", "WebGLShadow", {
						writes: [
							this._write("shadow:atlas", "framebuffer-depth"),
							this._write("shadow:transmittance", "framebuffer-color", true),
							this._write("shadow:particle-volume", "copy-target", true),
						],
					}),
				],
			],
			[
				"main-opaque",
				(pass) => [
					this._node(pass, "opaque-depth-prepass", "WebGLOpaqueDepthPrepass", {
						writes: [this._write("frame:depth", "framebuffer-depth")],
					}),
					this._node(pass, "opaque-scene", "WebGLOpaqueScene", {
						reads: [
							this._read("frame:depth", "framebuffer-depth", true),
							this._read("shadow:atlas", "texture-sampling", true),
							this._read("shadow:transmittance", "texture-sampling", true),
							this._read("shadow:particle-volume", "texture-sampling", true),
						],
						writes: [
							this._write("frame:scene-color", "framebuffer-color"),
							this._write("frame:motion-depth", "framebuffer-color"),
							this._write("frame:normal", "framebuffer-color", true),
							this._write("frame:depth", "framebuffer-depth"),
						],
					}),
				],
			],
			[
				"main-transparent",
				(pass, _context, state) =>
					state.oitActive ?
						this._createOITTransparentNodes(pass, state)
					:	this._createLegacyTransparentNodes(pass, _context),
			],
			[
				"particles",
				(pass, _context, state) =>
					state.oitActive ?
						this._createOITParticleNodes(pass)
					:	[
							this._node(pass, "particles", "WebGLParticles", {
								reads: [
									this._read("frame:depth", "framebuffer-depth", true),
									this._read("shadow:atlas", "texture-sampling", true),
								],
								writes: [
									this._write("frame:scene-color", "framebuffer-color"),
									this._write("frame:depth", "framebuffer-depth", true),
								],
							}),
						],
			],
		]);
	}

	private _createLegacyTransparentNodes(
		pass: FramePass,
		context: FrameContext,
	): WebGLFrameGraphNode[] {
		const packets = context.scene?.transparentPackets ?? [];
		if (!packets.some((packet) => materialUsesTransmission(packet.submission.material.effective))) {
			return [this._node(
				pass,
				"transparent-legacy",
				"WebGLLegacyTransparent",
				this._createTransparentResources(),
			)];
		}

		const nodes: WebGLFrameGraphNode[] = [this._node(
			pass,
			"transmission-depth-copy",
			"WebGLTransmissionOpaqueDepthCopy",
			{
				reads: [this._read("frame:motion-depth", "texture-sampling")],
				writes: [this._write(
					"frame:transmission-linear-depth",
					"framebuffer-color",
				)],
			},
		)];
		let segmentStart = 0;
		for (let index = 0; index < packets.length; index++) {
			if (
				!materialUsesTransmission(
					packets[index].submission.material.effective,
				)
			) continue;
			if (segmentStart < index) {
				nodes.push(this._node(
					pass,
					"transparent-legacy",
					`WebGLLegacyTransparentSegment${segmentStart}-${index}`,
					{
						...this._createTransparentResources(),
						packetStart: segmentStart,
						packetEnd: index,
					},
				));
			}
			nodes.push(
				this._node(
					pass,
					"transmission-background-copy",
					`WebGLTransmissionBackgroundCopy${index}`,
					{
						reads: [this._read("frame:scene-color", "texture-sampling")],
						writes: [this._write(
							"frame:transmission-background",
							"framebuffer-color",
						)],
						packetIndex: index,
					},
				),
				this._node(
					pass,
					"transmission-draw",
					`WebGLTransmissionDraw${index}`,
					{
						reads: [
							this._read("frame:transmission-background", "texture-sampling"),
							this._read("frame:transmission-linear-depth", "texture-sampling"),
							...this._createTransparentResources().reads ?? [],
						],
						writes: this._createTransparentResources().writes,
						packetIndex: index,
					},
				),
			);
			segmentStart = index + 1;
		}
		if (segmentStart < packets.length) {
			nodes.push(this._node(
				pass,
				"transparent-legacy",
				`WebGLLegacyTransparentSegment${segmentStart}-${packets.length}`,
				{
					...this._createTransparentResources(),
					packetStart: segmentStart,
					packetEnd: packets.length,
				},
			));
		}
		return nodes;
	}

	private _createOITTransparentNodes(
		pass: FramePass,
		state: WebGLFrameGraphPlannerState
	): WebGLFrameGraphNode[] {
		const nodes = [
			this._node(pass, "oit-clear", "WebGLOITTransparentClear", {
				scope: "transparent",
				requires: [
					{ id: "oit:accum" },
					{ id: "oit:reveal" },
				],
				writes: [
					this._write("oit:accum", "framebuffer-color"),
					this._write("oit:reveal", "framebuffer-color"),
				],
			}),
			this._node(pass, "oit-accum", "WebGLOITTransparentAccum", {
				scope: "transparent",
				requires: [{ id: "oit:accum" }],
				reads: [
					this._read("frame:depth", "framebuffer-depth", true),
					this._read("shadow:atlas", "texture-sampling", true),
					this._read("shadow:transmittance", "texture-sampling", true),
					this._read("shadow:particle-volume", "texture-sampling", true),
				],
				writes: [this._write("oit:accum", "framebuffer-color")],
			}),
			this._node(pass, "oit-reveal", "WebGLOITTransparentReveal", {
				scope: "transparent",
				requires: [{ id: "oit:reveal" }],
				reads: [
					this._read("frame:depth", "framebuffer-depth", true),
					this._read("shadow:atlas", "texture-sampling", true),
					this._read("shadow:transmittance", "texture-sampling", true),
					this._read("shadow:particle-volume", "texture-sampling", true),
				],
				writes: [this._write("oit:reveal", "framebuffer-color")],
			}),
		];
		if (!state.hasParticleSystems) {
			nodes.push(
				this._node(
					pass,
					"oit-copy-scene-color",
					"WebGLOITTransparentSceneColorCopy",
					{
						scope: "transparent",
						reads: [
							this._read("frame:scene-color", "texture-sampling"),
						],
						writes: [
							this._write("post:color", "framebuffer-color"),
						],
					},
				),
				this._node(pass, "oit-resolve", "WebGLOITTransparentResolve", {
					scope: "transparent",
					reads: [
						this._read("post:color", "texture-sampling"),
						this._read("oit:accum", "texture-sampling"),
						this._read("oit:reveal", "texture-sampling"),
					],
					writes: [
						this._write("frame:scene-color", "framebuffer-color"),
					],
				}),
				this._node(pass, "transparent-legacy", "WebGLOITLegacyTransparent", {
					scope: "transparent",
					reads: [
						this._read("frame:depth", "framebuffer-depth", true),
						this._read("shadow:atlas", "texture-sampling", true),
						this._read("shadow:transmittance", "texture-sampling", true),
						this._read("shadow:particle-volume", "texture-sampling", true),
					],
					writes: [
						this._write("frame:scene-color", "framebuffer-color"),
						this._write("frame:depth", "framebuffer-depth", true),
					],
				})
			);
		}
		return nodes;
	}

	private _createOITParticleNodes(pass: FramePass): WebGLFrameGraphNode[] {
		return [
			this._node(pass, "oit-clear", "WebGLOITParticleClear", {
				scope: "particles",
				requires: [
					{ id: "oit:accum" },
					{ id: "oit:reveal" },
				],
				writes: [
					this._write("oit:accum", "framebuffer-color"),
					this._write("oit:reveal", "framebuffer-color"),
				],
			}),
			this._node(pass, "oit-accum", "WebGLOITParticleAccum", {
				scope: "particles",
				requires: [{ id: "oit:accum" }],
				writes: [this._write("oit:accum", "framebuffer-color")],
			}),
			this._node(pass, "oit-reveal", "WebGLOITParticleReveal", {
				scope: "particles",
				requires: [{ id: "oit:reveal" }],
				writes: [this._write("oit:reveal", "framebuffer-color")],
			}),
			this._node(
				pass,
				"oit-copy-scene-color",
				"WebGLOITParticleSceneColorCopy",
				{
					scope: "particles",
					reads: [
						this._read("frame:scene-color", "texture-sampling"),
					],
					writes: [
						this._write("post:color", "framebuffer-color"),
					],
				},
			),
			this._node(pass, "oit-resolve", "WebGLOITParticleResolve", {
				scope: "particles",
				reads: [
					this._read("post:color", "texture-sampling"),
					this._read("oit:accum", "texture-sampling"),
					this._read("oit:reveal", "texture-sampling"),
				],
				writes: [
					this._write("frame:scene-color", "framebuffer-color"),
				],
			}),
			this._node(pass, "transparent-legacy", "WebGLOITParticleLegacyTransparent", {
				scope: "particles",
				writes: [this._write("frame:scene-color", "framebuffer-color")],
			}),
			this._node(pass, "particles", "WebGLParticlesAdditive", {
				scope: "particles",
				writes: [this._write("frame:scene-color", "framebuffer-color")],
			}),
		];
	}

	private _createTransparentResources(): Pick<
		WebGLFrameGraphNode,
		"reads" | "writes"
	> {
		return {
			reads: [
				this._read("frame:depth", "framebuffer-depth", true),
				this._read("shadow:atlas", "texture-sampling", true),
				this._read("shadow:transmittance", "texture-sampling", true),
				this._read("shadow:particle-volume", "texture-sampling", true),
			],
			writes: [
				this._write("frame:scene-color", "framebuffer-color"),
				this._write("frame:depth", "framebuffer-depth", true),
			],
		};
	}

	private _node(
		pass: FramePass,
		kind: WebGLFrameGraphNode["kind"],
		label: string,
		resources: Pick<
			WebGLFrameGraphNode,
			"creates" | "requires" | "reads" | "writes" | "destroys" | "scope" |
				"packetStart" | "packetEnd" | "packetIndex"
		> = {}
	): WebGLFrameGraphNode {
		const packetSuffix = resources.packetIndex !== undefined ?
			`:${resources.packetIndex}`
		:	resources.packetStart !== undefined ?
				`:${resources.packetStart}-${resources.packetEnd}`
			:	"";
		return {
			id: `${pass.stage}:${kind}:${resources.scope ?? "frame"}${packetSuffix}`,
			stage: pass.stage,
			kind,
			label,
			...resources,
		};
	}

	private _read(
		id: string,
		usage: WebGLFrameGraphResourceRef["usage"],
		optional = false
	): WebGLFrameGraphResourceRef {
		return { id, usage, optional };
	}

	private _write(
		id: string,
		usage: WebGLFrameGraphResourceRef["usage"],
		optional = false
	): WebGLFrameGraphResourceRef {
		return { id, usage, optional };
	}
}
