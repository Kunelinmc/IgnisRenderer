import type { Node } from "../core/Node";
import { AnimationAction } from "./AnimationAction";
import { AnimationClip } from "./AnimationClip";
import { AnimationLayer } from "./AnimationLayer";
import { AnimationStateMachine } from "./AnimationStateMachine";
import { BlendTree1D } from "./BlendTree1D";
import { BlendTreeDirect } from "./BlendTreeDirect";
import type { AnimationRootMotionOptions } from "./types";

export interface AnimationMixerOptions {
	root: Node;
	name?: string;
	rootMotion?: AnimationRootMotionOptions;
}

export class AnimationMixer {
	public readonly name: string;
	public readonly root: Node;
	public readonly layers: AnimationLayer[];
	public readonly clips: Map<string, AnimationClip>;
	public readonly stateMachines: Map<string, AnimationStateMachine>;
	public readonly blendTrees1D: Map<string, BlendTree1D>;
	public readonly blendTreesDirect: Map<string, BlendTreeDirect>;
	public readonly nodeBindings: Map<string, Node>;
	public readonly entityBindings: Map<string, number>;
	public readonly materialBindings: Map<string, any>;
	public readonly morphBindings: Map<string, any>;
	public rootMotion: Required<AnimationRootMotionOptions>;

	constructor(options: AnimationMixerOptions) {
		this.name = options.name ?? `${options.root.name}-mixer`;
		this.root = options.root;
		this.layers = [new AnimationLayer({ name: "Base Layer" })];
		this.clips = new Map();
		this.stateMachines = new Map();
		this.blendTrees1D = new Map();
		this.blendTreesDirect = new Map();
		this.nodeBindings = new Map();
		this.entityBindings = new Map();
		this.materialBindings = new Map();
		this.morphBindings = new Map();
		this.rootMotion = {
			enabled: options.rootMotion?.enabled ?? false,
			trackPath: options.rootMotion?.trackPath ?? options.root.name,
		};
	}

	public addClip(clip: AnimationClip): this {
		this.clips.set(clip.name, clip);
		return this;
	}

	public getClip(name: string): AnimationClip | null {
		return this.clips.get(name) ?? null;
	}

	public getOrCreateLayer(name: string): AnimationLayer {
		const existing = this.layers.find((layer) => layer.name === name);
		if (existing) return existing;
		const layer = new AnimationLayer({ name });
		this.layers.push(layer);
		return layer;
	}

	public getLayer(name: string): AnimationLayer | null {
		return this.layers.find((layer) => layer.name === name) ?? null;
	}

	public clipAction(
		clipName: string,
		layerName: string = "Base Layer"
	): AnimationAction {
		const clip = this.clips.get(clipName);
		if (!clip) {
			throw new Error(
				`AnimationMixer "${this.name}" missing clip "${clipName}"`
			);
		}
		const layer = this.getOrCreateLayer(layerName);
		const existing = layer.getAction(clipName);
		if (existing) return existing;
		const action = new AnimationAction(clip);
		layer.addAction(action);
		return action;
	}

	public bindNode(path: string, node: Node): void {
		this.nodeBindings.set(path, node);
	}

	public bindEntity(path: string, entityId: number): void {
		this.entityBindings.set(path, entityId);
	}

	public bindMaterial(path: string, material: any): void {
		this.materialBindings.set(path, material);
	}

	public bindMorph(path: string, binding: any): void {
		this.morphBindings.set(path, binding);
	}

	public addStateMachine(
		layerName: string,
		stateMachine: AnimationStateMachine
	): void {
		this.stateMachines.set(layerName, stateMachine);
	}

	public addBlendTree1D(tree: BlendTree1D): void {
		this.blendTrees1D.set(tree.name, tree);
	}

	public addBlendTreeDirect(tree: BlendTreeDirect): void {
		this.blendTreesDirect.set(tree.name, tree);
	}

	public hasActiveActions(): boolean {
		for (const layer of this.layers) {
			if (layer.hasActiveActions()) return true;
		}
		return false;
	}
}
