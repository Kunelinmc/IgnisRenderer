import type {
	IPostProcessExecutor,
	PostProcessResourceHandle,
	PostProcessTransientDescriptor,
	PostProcessTransientSlot,
	PostProcessTransientSlots,
} from "./types";
import {
	createPostProcessResourceAllocationKey,
	resolvePostProcessResourceDescriptor,
} from "./resourceDescriptors";

interface TransientEntry {
	descriptorKey: string;
	handle: PostProcessResourceHandle;
}

export interface PostProcessTransientPrepareRequest {
	readonly executor: IPostProcessExecutor;
	readonly descriptors: readonly PostProcessTransientDescriptor[];
	readonly width: number;
	readonly height: number;
}

export interface PostProcessTransientPrepareResult {
	readonly slots: PostProcessTransientSlots;
	readonly changed: boolean;
}

/**
 * Owns single-frame post-process transient resource handles.
 */
export class PostProcessTransientManager {
	private _entries = new Map<string, TransientEntry>();

	/**
	 * Ensures the requested transient resources exist for the current frame.
	 *
	 * @param request Frame transient allocation request.
	 * @returns Current transient slots and whether any resource changed.
	 * @sideEffects Allocates, destroys, or reuses transient resources.
	 */
	public prepare(
		request: PostProcessTransientPrepareRequest
	): PostProcessTransientPrepareResult {
		const activeIds = new Set<string>();
		let changed = false;
		for (const descriptor of request.descriptors) {
			activeIds.add(descriptor.id);
			changed = this._ensureEntry(descriptor, request) || changed;
		}
		for (const id of Array.from(this._entries.keys())) {
			if (!activeIds.has(id)) {
				this._destroyEntry(id, request.executor);
				changed = true;
			}
		}
		return {
			slots: this.getSlots(),
			changed,
		};
	}

	/**
	 * Returns current transient slots without mutating them.
	 *
	 * @returns Snapshot of active transient slots.
	 * @sideEffects None.
	 */
	public getSlots(): PostProcessTransientSlots {
		const slots: PostProcessTransientSlots = {};
		for (const [id, entry] of this._entries.entries()) {
			slots[id] = this._toSlot(id, entry);
		}
		return slots;
	}

	/**
	 * Destroys all active transient resources.
	 *
	 * @param executor Executor that owns the concrete resources.
	 * @returns Nothing.
	 * @sideEffects Calls `executor.destroyResource` for all transient handles.
	 */
	public destroy(executor: IPostProcessExecutor): void {
		for (const id of Array.from(this._entries.keys())) {
			this._destroyEntry(id, executor);
		}
	}

	private _ensureEntry(
		descriptor: PostProcessTransientDescriptor,
		request: PostProcessTransientPrepareRequest
	): boolean {
		const resourceDescriptor = resolvePostProcessResourceDescriptor(
			descriptor,
			request.width,
			request.height,
			{
				includeMipMode: true,
			}
		);
		const descriptorKey = createPostProcessResourceAllocationKey(
			request.executor.backend,
			resourceDescriptor,
			{
				includeMipMode: true,
				sortUsage: true,
			}
		);
		const current = this._entries.get(descriptor.id);
		if (current && current.descriptorKey === descriptorKey) {
			return false;
		}
		if (current) {
			request.executor.destroyResource(current.handle);
		}
		this._entries.set(descriptor.id, {
			descriptorKey,
			handle: request.executor.createResource(resourceDescriptor),
		});
		return true;
	}

	private _toSlot(id: string, entry: TransientEntry): PostProcessTransientSlot {
		return {
			id,
			handle: entry.handle,
		};
	}

	private _destroyEntry(id: string, executor: IPostProcessExecutor): void {
		const entry = this._entries.get(id);
		if (!entry) {
			return;
		}
		executor.destroyResource(entry.handle);
		this._entries.delete(id);
	}
}
