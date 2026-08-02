declare const RENDER_GRAPH_RESOURCE_ID: unique symbol;
declare const RENDER_GRAPH_NODE_ID: unique symbol;
declare const RENDER_GRAPH_PHYSICAL_RESOURCE_ID: unique symbol;

/** @internal Backend-private logical render graph resource identifier. */
export type RenderGraphResourceId = string & {
	readonly [RENDER_GRAPH_RESOURCE_ID]: true;
};

/** @internal Backend-private logical render graph node identifier. */
export type RenderGraphNodeId = string & {
	readonly [RENDER_GRAPH_NODE_ID]: true;
};

/** @internal Backend-private stable physical resource identifier. */
export type RenderGraphPhysicalResourceId = string & {
	readonly [RENDER_GRAPH_PHYSICAL_RESOURCE_ID]: true;
};

/** @internal Brands a backend-private logical render graph resource identifier. */
export function renderGraphResourceId(value: string): RenderGraphResourceId {
	return value as RenderGraphResourceId;
}

/** @internal Brands a backend-private logical render graph node identifier. */
export function renderGraphNodeId(value: string): RenderGraphNodeId {
	return value as RenderGraphNodeId;
}

/** @internal Brands a backend-private stable physical resource identifier. */
export function renderGraphPhysicalResourceId(value: string): RenderGraphPhysicalResourceId {
	return value as RenderGraphPhysicalResourceId;
}
