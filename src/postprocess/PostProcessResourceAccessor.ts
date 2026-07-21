import type {
	LogicalGBufferSemantic,
	PostProcessExecutionDeclaration,
	PostProcessNativeHistorySlot,
	PostProcessResourceAccessor,
} from "./types";

export interface PostProcessResourceAccessorOptions<TResource> {
	readonly passId: string;
	readonly declaration: PostProcessExecutionDeclaration;
	readonly colorInput: TResource | null;
	readonly colorOutput: TResource | null;
	readonly getGBuffer: (semantic: LogicalGBufferSemantic) => TResource | null;
	readonly getHistory: (id: string) => PostProcessNativeHistorySlot<TResource> | null;
	readonly getTransient: (id: string) => TResource | null;
	readonly getShared: (id: string) => TResource | null;
	readonly copyGBufferToHistory?: (
		semantic: LogicalGBufferSemantic,
		historyId: string
	) => void | Promise<void>;
}

/** @internal Creates a fixed accessor guarded by one execution declaration. */
export function createPostProcessResourceAccessor<TResource>(
	options: PostProcessResourceAccessorOptions<TResource>
): PostProcessResourceAccessor<TResource> {
	const gBuffer = new Map(
		(options.declaration.gBuffer ?? []).map((entry) => [entry.semantic, entry]),
	);
	const histories = new Map(
		(options.declaration.histories ?? []).map((entry) => [entry.descriptor.id, entry]),
	);
	const transients = new Map(
		(options.declaration.transients ?? []).map((entry) => [entry.descriptor.id, entry]),
	);
	const shared = new Map(
		(options.declaration.shared ?? []).map((entry) => [entry.id, entry]),
	);
	const undeclared = (kind: string, id: string): never => {
		throw new Error(
			`Post-process pass "${options.passId}" accessed undeclared ${kind} "${id}".`,
		);
	};
	return Object.freeze({
		color: Object.freeze({ input: options.colorInput, output: options.colorOutput }),
		getGBuffer: (semantic) => {
			const declaration = gBuffer.get(semantic);
			if (!declaration) return undeclared("G-buffer resource", semantic);
			const resource = options.getGBuffer(semantic);
			if (resource === null && declaration.optional !== true) {
				throw new Error(
					`Post-process pass "${options.passId}" is missing required G-buffer resource "${semantic}".`,
				);
			}
			return resource;
		},
		getHistory: (id) => {
			if (!histories.has(id)) return undeclared("history", id);
			const slot = options.getHistory(id);
			if (!slot) {
				throw new Error(`Post-process pass "${options.passId}" is missing history "${id}".`);
			}
			return slot;
		},
		getTransient: (id) => {
			if (!transients.has(id)) return undeclared("transient", id);
			const resource = options.getTransient(id);
			if (resource === null) {
				throw new Error(`Post-process pass "${options.passId}" is missing transient "${id}".`);
			}
			return resource;
		},
		getShared: (id) => {
			const declaration = shared.get(id);
			if (!declaration) return undeclared("shared resource", id);
			const resource = options.getShared(id);
			if (resource === null && declaration.optional !== true) {
				throw new Error(
					`Post-process pass "${options.passId}" is missing required shared resource "${id}".`,
				);
			}
			return resource;
		},
		copyGBufferToHistory: (semantic, historyId) => {
			if (!gBuffer.has(semantic)) return undeclared("G-buffer resource", semantic);
			const history = histories.get(historyId);
			if (!history) return undeclared("history", historyId);
			if (history.write.length === 0) {
				throw new Error(
					`Post-process pass "${options.passId}" cannot copy into read-only history "${historyId}".`,
				);
			}
			if (!options.copyGBufferToHistory) {
				throw new Error(
					`Post-process pass "${options.passId}" backend cannot copy G-buffer history.`,
				);
			}
			return options.copyGBufferToHistory(semantic, historyId);
		},
	});
}
