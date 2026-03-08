import { AnimationAction } from "./AnimationAction";
import type { AnimationLayerMask } from "./types";

export type AnimationLayerBlendMode = "override" | "additive";

export interface AnimationLayerOptions {
	name?: string;
	weight?: number;
	blendMode?: AnimationLayerBlendMode;
	mask?: AnimationLayerMask | null;
}

export class AnimationLayer {
	public readonly name: string;
	public weight: number;
	public blendMode: AnimationLayerBlendMode;
	public mask: AnimationLayerMask | null;
	private _actions: Map<string, AnimationAction>;

	constructor(options: AnimationLayerOptions = {}) {
		this.name = options.name ?? "Base Layer";
		this.weight = options.weight ?? 1;
		this.blendMode = options.blendMode ?? "override";
		this.mask = options.mask ?? null;
		this._actions = new Map();
	}

	public get actions(): AnimationAction[] {
		return Array.from(this._actions.values());
	}

	public getAction(name: string): AnimationAction | null {
		return this._actions.get(name) ?? null;
	}

	public addAction(action: AnimationAction): AnimationAction {
		this._actions.set(action.clip.name, action);
		return action;
	}

	public removeAction(name: string): boolean {
		return this._actions.delete(name);
	}

	public clearActions(): void {
		this._actions.clear();
	}

	public hasActiveActions(): boolean {
		for (const action of this._actions.values()) {
			if (action.enabled && action.weight > 0 && !action.finished) {
				return true;
			}
		}
		return false;
	}

	public allowsPath(path: string): boolean {
		if (!this.mask) return true;
		const include = this.mask.include ?? [];
		const exclude = this.mask.exclude ?? [];
		if (
			exclude.some((prefix) => path === prefix || path.startsWith(`${prefix}/`))
		) {
			return false;
		}
		if (include.length === 0) return true;
		return include.some(
			(prefix) => path === prefix || path.startsWith(`${prefix}/`)
		);
	}
}
