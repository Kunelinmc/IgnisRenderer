import type { MeshAsset } from "../meshes";
import { buildCSGMeshAsset } from "./solvers";
import type {
	CSGBuildOptions,
	CSGGraph,
	CSGLeafNode,
	CSGOperand,
	CSGOperation,
	CSGOperationNode,
	CSGRebuildResult,
} from "./types";

export type CSGGraphInput = CSGOperand | CSGGraph | CSGBuilder;

export class CSGBuilder {
	private _graph: CSGGraph;

	constructor(input: CSGGraphInput) {
		this._graph = normalizeGraphInput(input);
	}

	public getGraph(): CSGGraph {
		return cloneGraph(this._graph);
	}

	public setGraph(input: CSGGraphInput): this {
		this._graph = normalizeGraphInput(input);
		return this;
	}

	public union(input: CSGGraphInput): this {
		return this._compose("union", input);
	}

	public subtract(input: CSGGraphInput): this {
		return this._compose("subtract", input);
	}

	public intersect(input: CSGGraphInput): this {
		return this._compose("intersect", input);
	}

	public xor(input: CSGGraphInput): this {
		return this._compose("xor", input);
	}

	public solve(options: CSGBuildOptions = {}): CSGRebuildResult {
		return buildCSGMeshAsset(this._graph, options);
	}

	public toMeshAsset(options: CSGBuildOptions = {}): MeshAsset {
		const result = this.solve(options);
		if (result.ok && result.meshAsset) {
			return result.meshAsset;
		}
		const reason =
			result.diagnostics[0]?.message ??
			"CSG build failed with an unknown error";
		throw new Error(reason);
	}

	private _compose(operation: CSGOperation, input: CSGGraphInput): this {
		const next: CSGOperationNode = {
			kind: "op",
			operation,
			left: this.getGraph(),
			right: normalizeGraphInput(input),
		};
		this._graph = next;
		return this;
	}
}

export class CSG {
	public static from(input: CSGGraphInput): CSGBuilder {
		return new CSGBuilder(input);
	}
}

export function isCSGGraph(value: unknown): value is CSGGraph {
	if (!value || typeof value !== "object") return false;
	const candidate = value as {
		kind?: unknown;
		operation?: unknown;
		left?: unknown;
		right?: unknown;
		operand?: unknown;
	};
	if (candidate.kind === "leaf") {
		return "operand" in candidate;
	}
	if (candidate.kind === "op") {
		return (
			(candidate.operation === "union" ||
				candidate.operation === "subtract" ||
				candidate.operation === "intersect" ||
				candidate.operation === "xor") &&
			"left" in candidate &&
			"right" in candidate
		);
	}
	return false;
}

export function cloneGraph(graph: CSGGraph): CSGGraph {
	if (graph.kind === "leaf") {
		return {
			kind: "leaf",
			operand: graph.operand,
		};
	}
	return {
		kind: "op",
		operation: graph.operation,
		left: cloneGraph(graph.left),
		right: cloneGraph(graph.right),
	};
}

export function normalizeGraphInput(input: CSGGraphInput): CSGGraph {
	if (input instanceof CSGBuilder) {
		return input.getGraph();
	}
	if (isCSGGraph(input)) {
		return cloneGraph(input);
	}
	return {
		kind: "leaf",
		operand: input as CSGOperand,
	};
}
