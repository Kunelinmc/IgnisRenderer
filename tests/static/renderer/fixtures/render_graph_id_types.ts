import {
	renderGraphNodeId,
	renderGraphPhysicalResourceId,
	renderGraphResourceId,
} from "../../../../src/rendergraph/ids.ts";
import type {
	RenderGraphNodeId,
	RenderGraphPhysicalResourceId,
	RenderGraphResourceId,
} from "../../../../src/rendergraph/ids.ts";

const resourceId = renderGraphResourceId("resource");
const nodeId = renderGraphNodeId("node");
const physicalId = renderGraphPhysicalResourceId("physical");

const resourceString: string = resourceId;
const nodeString: string = nodeId;
const physicalString: string = physicalId;

// @ts-expect-error Ordinary strings must enter through the matching constructor.
const unbrandedResourceId: RenderGraphResourceId = "resource";
// @ts-expect-error Logical resource and node IDs must not be interchangeable.
const resourceAsNodeId: RenderGraphNodeId = resourceId;
// @ts-expect-error Node and physical resource IDs must not be interchangeable.
const nodeAsPhysicalId: RenderGraphPhysicalResourceId = nodeId;
// @ts-expect-error Physical and logical resource IDs must not be interchangeable.
const physicalAsResourceId: RenderGraphResourceId = physicalId;

void resourceString;
void nodeString;
void physicalString;
void unbrandedResourceId;
void resourceAsNodeId;
void nodeAsPhysicalId;
void physicalAsResourceId;
