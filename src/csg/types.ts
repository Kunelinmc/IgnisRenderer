import type { Material } from "../materials/Material";
import type { Matrix4 } from "../maths/Matrix4";
import type { MeshAsset, MeshInstance } from "../meshes";

export type CSGOperation = "union" | "subtract" | "intersect" | "xor";

export type CSGSolverPreference = "auto" | "builtin" | "wasm";

export type CSGExecutionMode = "sync" | "worker";

export type CSGPhysicsSyncMode = "off" | "auto";

export interface CSGOperandDescriptor {
	mesh: MeshAsset | MeshInstance;
	transform?: Matrix4 | null;
	material?: Material | null;
	label?: string;
}

export type CSGOperand = MeshAsset | MeshInstance | CSGOperandDescriptor;

export interface CSGLeafNode {
	kind: "leaf";
	operand: CSGOperand;
}

export interface CSGOperationNode {
	kind: "op";
	operation: CSGOperation;
	left: CSGGraph;
	right: CSGGraph;
}

export type CSGGraph = CSGLeafNode | CSGOperationNode;

export type CSGDiagnosticSeverity = "error" | "warning" | "info";

export interface CSGDiagnostic {
	code: string;
	message: string;
	severity: CSGDiagnosticSeverity;
	context?: Record<string, unknown>;
}

export interface CSGBuildOptions {
	epsilon?: number;
	maxOutputTriangles?: number;
	capMaterial?: Material | null;
	solverPreference?: CSGSolverPreference;
}

export interface CSGResolvedBuildOptions {
	epsilon: number;
	maxOutputTriangles: number;
	capMaterial: Material | null;
	solverPreference: CSGSolverPreference;
}

export interface CSGRebuildResult {
	ok: boolean;
	meshAsset: MeshAsset | null;
	triangleCount: number;
	solverId: string;
	fallbackUsed: boolean;
	stale: boolean;
	diagnostics: CSGDiagnostic[];
}

export interface CSGSolveRequest {
	graph: CSGGraph;
	options: CSGResolvedBuildOptions;
}

export interface CSGSolveResult {
	meshAsset: MeshAsset;
	triangleCount: number;
	diagnostics?: CSGDiagnostic[];
}

export interface CSGSolverAdapter {
	id: string;
	solve(request: CSGSolveRequest): CSGSolveResult;
}
