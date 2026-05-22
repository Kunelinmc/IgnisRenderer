import type {
	IPostProcessExecutor,
	PostProcessHistoryDescriptor,
	PostProcessHistorySlot,
	PostProcessHistorySlots,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "./types";

interface HistoryEntry {
	descriptorKey: string;
	read: PostProcessResourceHandle;
	write: PostProcessResourceHandle;
	valid: boolean;
	updated: boolean;
}

export interface PostProcessHistoryPrepareRequest {
	readonly executor: IPostProcessExecutor;
	readonly descriptors: readonly PostProcessHistoryDescriptor[];
	readonly width: number;
	readonly height: number;
	readonly reset: boolean;
	readonly signature: string;
}

/**
 * Owns temporal post-process history resource handles and validity state.
 */
export class PostProcessHistoryManager {
	private _entries = new Map<string, HistoryEntry>();
	private _lastSignature = "";

	/**
	 * Ensures the requested history resources exist for the current frame.
	 *
	 * @param request Frame history allocation and invalidation request.
	 * @returns Current read/write slots keyed by history id.
	 * @sideEffects Allocates, destroys, or invalidates history resources.
	 */
	public prepare(request: PostProcessHistoryPrepareRequest): PostProcessHistorySlots {
		const signatureChanged =
			this._lastSignature !== "" && this._lastSignature !== request.signature;
		this._lastSignature = request.signature;
		const activeIds = new Set<string>();
		for (const descriptor of request.descriptors) {
			activeIds.add(descriptor.id);
			this._ensureEntry(descriptor, request);
		}
		for (const id of Array.from(this._entries.keys())) {
			if (!activeIds.has(id)) {
				this._destroyEntry(id, request.executor);
			}
		}
		if (request.reset || signatureChanged) {
			this.invalidate();
		}
		return this.getSlots();
	}

	/**
	 * Returns current history slots without mutating them.
	 *
	 * @returns Snapshot of active history slots.
	 * @sideEffects None.
	 */
	public getSlots(): PostProcessHistorySlots {
		const slots: PostProcessHistorySlots = {};
		for (const [id, entry] of this._entries.entries()) {
			slots[id] = this._toSlot(id, entry);
		}
		return slots;
	}

	/**
	 * Marks one history resource as written during the frame.
	 *
	 * @param id History id.
	 * @returns Nothing.
	 * @sideEffects Marks the slot for read/write swap at frame end.
	 */
	public markUpdated(id: string): void {
		const entry = this._entries.get(id);
		if (entry) {
			entry.updated = true;
		}
	}

	/**
	 * Marks multiple history resources as written during the frame.
	 *
	 * @param ids History ids.
	 * @returns Nothing.
	 * @sideEffects Marks matching slots for read/write swap at frame end.
	 */
	public markUpdatedMany(ids: readonly string[]): void {
		for (const id of ids) {
			this.markUpdated(id);
		}
	}

	/**
	 * Invalidates one history slot or all slots.
	 *
	 * @param id Optional history id. Omit to invalidate every slot.
	 * @returns Nothing.
	 * @sideEffects Clears validity and pending update flags.
	 */
	public invalidate(id?: string): void {
		if (id) {
			const entry = this._entries.get(id);
			if (entry) {
				entry.valid = false;
				entry.updated = false;
			}
			return;
		}
		for (const entry of this._entries.values()) {
			entry.valid = false;
			entry.updated = false;
		}
	}

	/**
	 * Finalizes history state after pass execution.
	 *
	 * @returns Nothing.
	 * @sideEffects Swaps updated read/write handles and marks them valid.
	 */
	public endFrame(): void {
		for (const entry of this._entries.values()) {
			if (!entry.updated) {
				continue;
			}
			const read = entry.read;
			entry.read = entry.write;
			entry.write = read;
			entry.valid = true;
			entry.updated = false;
		}
	}

	/**
	 * Destroys all active history resources.
	 *
	 * @param executor Executor that owns the concrete resources.
	 * @returns Nothing.
	 * @sideEffects Calls `executor.destroyResource` for all history handles.
	 */
	public destroy(executor: IPostProcessExecutor): void {
		for (const id of Array.from(this._entries.keys())) {
			this._destroyEntry(id, executor);
		}
		this._lastSignature = "";
	}

	private _ensureEntry(
		descriptor: PostProcessHistoryDescriptor,
		request: PostProcessHistoryPrepareRequest
	): void {
		const resourceDescriptor = this._toResourceDescriptor(
			descriptor,
			request.width,
			request.height
		);
		const descriptorKey = [
			request.executor.backend,
			resourceDescriptor.width,
			resourceDescriptor.height,
			resourceDescriptor.format,
			resourceDescriptor.usage.join(","),
		].join("|");
		const current = this._entries.get(descriptor.id);
		if (current && current.descriptorKey === descriptorKey) {
			return;
		}
		if (current) {
			request.executor.destroyResource(current.read);
			request.executor.destroyResource(current.write);
		}
		this._entries.set(descriptor.id, {
			descriptorKey,
			read: request.executor.createResource({
				...resourceDescriptor,
				id: `${descriptor.id}:read`,
			}),
			write: request.executor.createResource({
				...resourceDescriptor,
				id: `${descriptor.id}:write`,
			}),
			valid: false,
			updated: false,
		});
	}

	private _toResourceDescriptor(
		descriptor: PostProcessHistoryDescriptor,
		width: number,
		height: number
	): PostProcessResourceDescriptor {
		const resolvedWidth = Math.max(
			1,
			Math.floor(width * (descriptor.widthScale ?? 1))
		);
		const resolvedHeight = Math.max(
			1,
			Math.floor(height * (descriptor.heightScale ?? 1))
		);
		return {
			id: descriptor.id,
			width: resolvedWidth,
			height: resolvedHeight,
			format: descriptor.format ?? "rgba16float",
			usage: descriptor.usage ?? ["sampled", "storage", "render-target"],
		};
	}

	private _toSlot(id: string, entry: HistoryEntry): PostProcessHistorySlot {
		return {
			id,
			read: entry.read,
			write: entry.write,
			valid: entry.valid,
		};
	}

	private _destroyEntry(id: string, executor: IPostProcessExecutor): void {
		const entry = this._entries.get(id);
		if (!entry) {
			return;
		}
		executor.destroyResource(entry.read);
		executor.destroyResource(entry.write);
		this._entries.delete(id);
	}
}
