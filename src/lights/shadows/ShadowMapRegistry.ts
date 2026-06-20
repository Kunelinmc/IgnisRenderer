import { CascadedShadowMap } from "./CascadedShadowMap";
import type { ShadowMapBase } from "./ShadowMapBase";
import { SingleShadowMap } from "./SingleShadowMap";
import { VarianceShadowMap, type VarianceShadowMapOptions } from "./VarianceShadowMap";
import type {
	CascadedShadowMapOptions,
	ShadowMapBaseOptions,
	ShadowMapKind,
} from "./types";

export type ShadowMapFactory<
	TShadowMap extends ShadowMapBase = ShadowMapBase,
	TOptions extends ShadowMapBaseOptions = ShadowMapBaseOptions,
> = (options?: TOptions) => TShadowMap;

/**
 * Registry for user-defined and built-in shadow map factories.
 */
export class ShadowMapRegistry {
	private readonly _factories = new Map<
		ShadowMapKind,
		ShadowMapFactory<ShadowMapBase, ShadowMapBaseOptions>
	>();

	/**
	 * Registers a factory for a shadow map kind.
	 *
	 * @param kind - Stable kind string used by `ShadowManager.create`.
	 * @param factory - Factory that receives creation options and returns a map.
	 * @returns This registry for chained registration.
	 * @throws If `kind` is an empty string.
	 * @sideEffects Replaces any existing factory registered for the same kind.
	 */
	public register<
		TShadowMap extends ShadowMapBase,
		TOptions extends ShadowMapBaseOptions = ShadowMapBaseOptions,
	>(
		kind: ShadowMapKind,
		factory: ShadowMapFactory<TShadowMap, TOptions>
	): this {
		if (!kind) {
			throw new Error("Shadow map kind must be a non-empty string.");
		}
		this._factories.set(
			kind,
			factory as ShadowMapFactory<ShadowMapBase, ShadowMapBaseOptions>
		);
		return this;
	}

	/**
	 * Removes a shadow map factory.
	 *
	 * @param kind - Kind string to remove.
	 * @returns `true` when a factory existed and was removed.
	 * @sideEffects Future `create` calls for the kind will fail unless re-registered.
	 */
	public unregister(kind: ShadowMapKind): boolean {
		return this._factories.delete(kind);
	}

	/**
	 * Checks whether a shadow map kind can be created.
	 *
	 * @param kind - Kind string to test.
	 * @returns `true` when a factory is registered for `kind`.
	 */
	public has(kind: ShadowMapKind): boolean {
		return this._factories.has(kind);
	}

	/**
	 * Creates a shadow map for a registered kind.
	 *
	 * @param kind - Kind string registered through `register`.
	 * @param options - Options forwarded to the registered factory.
	 * @returns The shadow map returned by the factory.
	 * @throws If no factory is registered for `kind`.
	 * @sideEffects Runs the factory and returns a new shadow map instance.
	 */
	public create<
		TShadowMap extends ShadowMapBase = ShadowMapBase,
		TOptions extends ShadowMapBaseOptions = ShadowMapBaseOptions,
	>(kind: ShadowMapKind, options?: TOptions): TShadowMap {
		const factory = this._factories.get(kind);
		if (!factory) {
			throw new Error(`Shadow map kind is not registered: ${kind}`);
		}
		return factory(options) as TShadowMap;
	}

	/**
	 * Copies the registry factory table.
	 *
	 * @returns A new registry with the same kind-to-factory mappings.
	 * @sideEffects None.
	 */
	public clone(): ShadowMapRegistry {
		const registry = new ShadowMapRegistry();
		for (const [kind, factory] of this._factories) {
			registry._factories.set(kind, factory);
		}
		return registry;
	}
}

/**
 * Creates a registry containing IgnisRenderer's built-in shadow map kinds.
 *
 * @returns A new registry with `single`, `vsm`, and `csm` factories.
 * @sideEffects None.
 */
export function createDefaultShadowMapRegistry(): ShadowMapRegistry {
	return new ShadowMapRegistry()
		.register<SingleShadowMap, ShadowMapBaseOptions>(
			"single",
			(options) => new SingleShadowMap(options)
		)
		.register<VarianceShadowMap, VarianceShadowMapOptions>(
			"vsm",
			(options) => new VarianceShadowMap(options)
		)
		.register<CascadedShadowMap, CascadedShadowMapOptions>(
			"csm",
			(options) => new CascadedShadowMap(options)
		);
}
