export interface PostProcessGraphNode {
	readonly id: string;
	readonly dependsOn?: readonly string[];
}

/**
 * Stores logical post-process nodes and resolves dependency order.
 */
export class PostProcessGraph<TNode extends PostProcessGraphNode> {
	private _nodes = new Map<string, TNode>();
	private _reservedIds = new Set<string>();

	constructor(nodes: readonly TNode[] = [], reservedIds: readonly string[] = []) {
		this._reservedIds = new Set(reservedIds);
		for (const node of nodes) {
			this.register(node, true);
		}
	}

	/**
	 * Registers a logical post-process graph node.
	 *
	 * @param node Node descriptor with a unique id.
	 * @param allowReserved Whether built-in reserved ids may be registered.
	 * @returns Nothing.
	 * @throws If the id is empty, reserved, or already registered.
	 * @sideEffects Mutates graph registration state.
	 */
	public register(node: TNode, allowReserved = false): void {
		this._assertCanRegister(node, allowReserved);
		this._nodes.set(node.id, node);
	}

	/**
	 * Removes a logical post-process graph node.
	 *
	 * @param id Custom node id to remove.
	 * @returns Nothing.
	 * @throws If `id` is reserved by a built-in pass.
	 * @sideEffects Mutates graph registration state.
	 */
	public unregister(id: string): void {
		if (this._reservedIds.has(id)) {
			throw new Error(`Cannot unregister built-in post-process pass "${id}".`);
		}
		this._nodes.delete(id);
	}

	/**
	 * Looks up a graph node by id.
	 *
	 * @param id Node id.
	 * @returns The registered node or `null`.
	 * @sideEffects None.
	 */
	public get(id: string): TNode | null {
		return this._nodes.get(id) ?? null;
	}

	/**
	 * Returns whether a graph node is registered.
	 *
	 * @param id Node id.
	 * @returns `true` when the graph contains `id`.
	 * @sideEffects None.
	 */
	public has(id: string): boolean {
		return this._nodes.has(id);
	}

	/**
	 * Returns all registered nodes in insertion order.
	 *
	 * @returns Copy of registered graph nodes.
	 * @sideEffects None.
	 */
	public values(): TNode[] {
		return Array.from(this._nodes.values());
	}

	/**
	 * Resolves enabled nodes into dependency order.
	 *
	 * @param isEnabled Predicate that selects nodes for this frame.
	 * @param warn Diagnostic sink for missing dependencies and cycles.
	 * @returns Enabled nodes in executable order, excluding invalid branches.
	 * @sideEffects Emits diagnostics through `warn`.
	 */
	public getExecutionOrder(
		isEnabled: (node: TNode) => boolean,
		warn: (key: string, message: string) => void
	): TNode[] {
		const enabled = new Map<string, TNode>();
		for (const [id, node] of this._nodes.entries()) {
			if (isEnabled(node)) {
				enabled.set(id, node);
			}
		}

		const order: TNode[] = [];
		const state = new Map<string, number>();
		const invalid = new Set<string>();

		const visit = (id: string): boolean => {
			if (invalid.has(id)) return false;
			const current = state.get(id) ?? 0;
			if (current === 2) return true;
			if (current === 1) {
				warn(
					`postprocess-cycle-${id}`,
					`Post-process dependency cycle detected at pass "${id}", skipping cycle branch`
				);
				invalid.add(id);
				return false;
			}

			const node = enabled.get(id);
			if (!node) return false;
			state.set(id, 1);

			for (const dependencyId of node.dependsOn ?? []) {
				if (!this._nodes.has(dependencyId)) {
					warn(
						`postprocess-dependency-missing-${id}-${dependencyId}`,
						`Post-process pass "${id}" depends on unknown pass "${dependencyId}"; skipping "${id}"`
					);
					invalid.add(id);
					state.set(id, 2);
					return false;
				}
				if (!enabled.has(dependencyId)) {
					continue;
				}
				if (!visit(dependencyId)) {
					invalid.add(id);
					state.set(id, 2);
					return false;
				}
			}

			state.set(id, 2);
			order.push(node);
			return true;
		};

		for (const id of enabled.keys()) {
			visit(id);
		}

		return order;
	}

	private _assertCanRegister(node: TNode, allowReserved: boolean): void {
		if (!node.id) {
			throw new Error("Post-process pass id is required.");
		}
		if (!allowReserved && this._reservedIds.has(node.id)) {
			throw new Error(
				`Cannot register built-in post-process pass "${node.id}".`
			);
		}
		if (this._nodes.has(node.id)) {
			throw new Error(
				`Post-process pass "${node.id}" is already registered.`
			);
		}
	}
}
