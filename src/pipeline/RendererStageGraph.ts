import { Logger } from "../foundation/Logger";

export interface RendererStageEnableContext {
	hasActiveAnimations: boolean;
	hasParticleSystems: boolean;
}

export interface RendererStageDefinition {
	id: string;
	dependsOn: string[];
	enabled?: (context: RendererStageEnableContext) => boolean;
}

export function resolveRendererStageOrder(
	stages: ReadonlyMap<string, RendererStageDefinition>,
	context: RendererStageEnableContext
): RendererStageDefinition[] {
	const enabled = new Map<string, RendererStageDefinition>();
	for (const [id, stage] of stages) {
		const isEnabled = stage.enabled ? stage.enabled(context) : true;
		if (!isEnabled) continue;
		enabled.set(id, stage);
	}

	const order: RendererStageDefinition[] = [];
	const state = new Map<string, number>();
	const invalid = new Set<string>();

	const visit = (id: string): boolean => {
		if (invalid.has(id)) return false;
		const current = state.get(id) ?? 0;
		if (current === 2) return true;
		if (current === 1) {
			const key = `renderer-stage-cycle-${id}`;
			Logger.warn(
				`[${key}] Renderer stage dependency cycle detected at "${id}", skipping cycle branch`,
				{ scope: "Renderer", onceKey: key },
			);
			invalid.add(id);
			return false;
		}

		const stage = enabled.get(id);
		if (!stage) return false;
		state.set(id, 1);

		for (const dependencyId of stage.dependsOn) {
			if (!stages.has(dependencyId)) {
				const key = `renderer-stage-dependency-missing-${id}-${dependencyId}`;
				Logger.warn(
					`[${key}] Renderer stage "${id}" depends on unknown stage ` +
						`"${dependencyId}", skipping "${id}"`,
					{ scope: "Renderer", onceKey: key },
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
		order.push({
			id: stage.id,
			dependsOn: [...stage.dependsOn],
			enabled: stage.enabled,
		});
		return true;
	};

	for (const id of enabled.keys()) {
		visit(id);
	}

	return order;
}
