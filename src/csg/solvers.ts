import { Material } from "../materials/Material";
import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import { MeshAsset, type MeshFace } from "../meshes";
import { MeshInstance } from "../meshes/MeshInstance";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import type {
	CSGBuildOptions,
	CSGDiagnostic,
	CSGGraph,
	CSGLeafNode,
	CSGOperand,
	CSGOperation,
	CSGOperandDescriptor,
	CSGRebuildResult,
	CSGResolvedBuildOptions,
	CSGSolveRequest,
	CSGSolveResult,
	CSGSolverAdapter,
	CSGSolverPreference,
} from "./types";

const DEFAULT_EPSILON = 1e-5;
const DEFAULT_MAX_OUTPUT_TRIANGLES = 200000;
const LARGE_VALUE = 1e9;

const _registeredWasmSolvers = new Map<string, CSGSolverAdapter>();

interface VertexPayload {
	position: Vector3;
	normal: Vector3;
	u: number;
	v: number;
}

interface PolygonShared {
	material: Material;
	source: string;
	generated: boolean;
}

interface OperandSolveState {
	materials: Material[];
}

interface SolveContext {
	options: CSGResolvedBuildOptions;
	diagnostics: CSGDiagnostic[];
	attributeDropWarningIssued: boolean;
	operandStateBySource: Map<string, OperandSolveState>;
}

class CSGBuildError extends Error {
	public readonly diagnostic: CSGDiagnostic;

	constructor(diagnostic: CSGDiagnostic) {
		super(diagnostic.message);
		this.diagnostic = diagnostic;
	}
}

class BSPVertex {
	public position: Vector3;
	public normal: Vector3;
	public u: number;
	public v: number;

	constructor(data: VertexPayload) {
		this.position = new Vector3(
			data.position.x,
			data.position.y,
			data.position.z
		);
		this.normal = normalizeVector(data.normal);
		this.u = data.u;
		this.v = data.v;
	}

	public clone(): BSPVertex {
		return new BSPVertex({
			position: this.position,
			normal: this.normal,
			u: this.u,
			v: this.v,
		});
	}

	public flip(): void {
		this.normal = scaleVector(this.normal, -1);
	}

	public interpolate(other: BSPVertex, t: number): BSPVertex {
		return new BSPVertex({
			position: addVector(
				scaleVector(this.position, 1 - t),
				scaleVector(other.position, t)
			),
			normal: normalizeVector(
				addVector(
					scaleVector(this.normal, 1 - t),
					scaleVector(other.normal, t)
				)
			),
			u: this.u + (other.u - this.u) * t,
			v: this.v + (other.v - this.v) * t,
		});
	}
}

class BSPPlane {
	public normal: Vector3;
	public w: number;
	private _epsilon: number;

	constructor(normal: Vector3, w: number, epsilon: number) {
		this.normal = normalizeVector(normal);
		this.w = w;
		this._epsilon = epsilon;
	}

	public clone(): BSPPlane {
		return new BSPPlane(this.normal, this.w, this._epsilon);
	}

	public flip(): void {
		this.normal = scaleVector(this.normal, -1);
		this.w = -this.w;
	}

	public static fromPolygon(
		polygon: BSPPolygon,
		epsilon: number
	): BSPPlane {
		const a = polygon.vertices[0].position;
		const b = polygon.vertices[1].position;
		const c = polygon.vertices[2].position;
		const normal = normalizeVector(
			crossVector(subVector(b, a), subVector(c, a))
		);
		return new BSPPlane(normal, dotVector(normal, a), epsilon);
	}

	public splitPolygon(
		polygon: BSPPolygon,
		coplanarFront: BSPPolygon[],
		coplanarBack: BSPPolygon[],
		front: BSPPolygon[],
		back: BSPPolygon[]
	): void {
		let polygonType = 0;
		const types: number[] = [];

		for (const vertex of polygon.vertices) {
			const delta = dotVector(this.normal, vertex.position) - this.w;
			const type =
				delta < -this._epsilon ? POLYGON_BACK
				: delta > this._epsilon ? POLYGON_FRONT
				: POLYGON_COPLANAR;
			polygonType |= type;
			types.push(type);
		}

		switch (polygonType) {
			case POLYGON_COPLANAR: {
				const destination =
					dotVector(this.normal, polygon.plane.normal) > 0 ?
						coplanarFront
					:	coplanarBack;
				destination.push(polygon.clone());
				break;
			}
			case POLYGON_FRONT:
				front.push(polygon.clone());
				break;
			case POLYGON_BACK:
				back.push(polygon.clone());
				break;
			default: {
				const frontVertices: BSPVertex[] = [];
				const backVertices: BSPVertex[] = [];
				for (let index = 0; index < polygon.vertices.length; index++) {
					const nextIndex = (index + 1) % polygon.vertices.length;
					const currentType = types[index];
					const nextType = types[nextIndex];
					const currentVertex = polygon.vertices[index];
					const nextVertex = polygon.vertices[nextIndex];

					if (currentType !== POLYGON_BACK) {
						frontVertices.push(currentVertex.clone());
					}
					if (currentType !== POLYGON_FRONT) {
						backVertices.push(currentVertex.clone());
					}

					if ((currentType | nextType) !== POLYGON_SPANNING) {
						continue;
					}

					const direction = subVector(
						nextVertex.position,
						currentVertex.position
					);
					const denominator = dotVector(this.normal, direction);
					const t =
						Math.abs(denominator) <= this._epsilon ? 0
						: (this.w -
								dotVector(this.normal, currentVertex.position)) /
							denominator;
					const intersection = currentVertex.interpolate(
						nextVertex,
						clamp01(t)
					);
					frontVertices.push(intersection);
					backVertices.push(intersection.clone());
				}

				if (frontVertices.length >= 3) {
					front.push(
						new BSPPolygon(frontVertices, {
							...polygon.shared,
							generated: true,
						}, this._epsilon)
					);
				}
				if (backVertices.length >= 3) {
					back.push(
						new BSPPolygon(backVertices, {
							...polygon.shared,
							generated: true,
						}, this._epsilon)
					);
				}
			}
		}
	}
}

class BSPPolygon {
	public vertices: BSPVertex[];
	public shared: PolygonShared;
	public plane: BSPPlane;
	private _epsilon: number;

	constructor(vertices: BSPVertex[], shared: PolygonShared, epsilon: number) {
		this.vertices = vertices;
		this.shared = shared;
		this._epsilon = epsilon;
		this.plane = BSPPlane.fromPolygon(this, epsilon);
	}

	public clone(): BSPPolygon {
		return new BSPPolygon(
			this.vertices.map((vertex) => vertex.clone()),
			{ ...this.shared },
			this._epsilon
		);
	}

	public flip(): void {
		this.vertices.reverse().forEach((vertex) => vertex.flip());
		this.plane.flip();
	}
}

class BSPNode {
	public plane: BSPPlane | null = null;
	public front: BSPNode | null = null;
	public back: BSPNode | null = null;
	public polygons: BSPPolygon[] = [];
	private _epsilon: number;

	constructor(polygons: BSPPolygon[] = [], epsilon: number = DEFAULT_EPSILON) {
		this._epsilon = epsilon;
		if (polygons.length > 0) {
			this.build(polygons);
		}
	}

	public clone(): BSPNode {
		const cloned = new BSPNode([], this._epsilon);
		cloned.plane = this.plane?.clone() ?? null;
		cloned.front = this.front?.clone() ?? null;
		cloned.back = this.back?.clone() ?? null;
		cloned.polygons = this.polygons.map((polygon) => polygon.clone());
		return cloned;
	}

	public invert(): void {
		for (const polygon of this.polygons) {
			polygon.flip();
		}
		this.plane?.flip();
		this.front?.invert();
		this.back?.invert();
		const nextFront = this.back;
		this.back = this.front;
		this.front = nextFront;
	}

	public clipPolygons(polygons: BSPPolygon[]): BSPPolygon[] {
		if (!this.plane) return polygons.slice();
		const front: BSPPolygon[] = [];
		const back: BSPPolygon[] = [];
		for (const polygon of polygons) {
			this.plane.splitPolygon(
				polygon,
				front,
				back,
				front,
				back
			);
		}
		const clippedFront =
			this.front ? this.front.clipPolygons(front) : front;
		const clippedBack =
			this.back ? this.back.clipPolygons(back) : [];
		return clippedFront.concat(clippedBack);
	}

	public clipTo(node: BSPNode): void {
		this.polygons = node.clipPolygons(this.polygons);
		this.front?.clipTo(node);
		this.back?.clipTo(node);
	}

	public allPolygons(): BSPPolygon[] {
		let result = this.polygons.slice();
		if (this.front) {
			result = result.concat(this.front.allPolygons());
		}
		if (this.back) {
			result = result.concat(this.back.allPolygons());
		}
		return result;
	}

	public build(polygons: BSPPolygon[]): void {
		if (polygons.length === 0) return;
		if (!this.plane) {
			this.plane = polygons[0].plane.clone();
		}

		const front: BSPPolygon[] = [];
		const back: BSPPolygon[] = [];
		for (const polygon of polygons) {
			this.plane.splitPolygon(
				polygon,
				this.polygons,
				this.polygons,
				front,
				back
			);
		}
		if (front.length > 0) {
			if (!this.front) this.front = new BSPNode([], this._epsilon);
			this.front.build(front);
		}
		if (back.length > 0) {
			if (!this.back) this.back = new BSPNode([], this._epsilon);
			this.back.build(back);
		}
	}
}

class BSPSolid {
	public polygons: BSPPolygon[];
	private _epsilon: number;

	constructor(polygons: BSPPolygon[], epsilon: number) {
		this.polygons = polygons;
		this._epsilon = epsilon;
	}

	public clone(): BSPSolid {
		return new BSPSolid(
			this.polygons.map((polygon) => polygon.clone()),
			this._epsilon
		);
	}

	public union(other: BSPSolid): BSPSolid {
		const leftNode = new BSPNode(this.clone().polygons, this._epsilon);
		const rightNode = new BSPNode(other.clone().polygons, this._epsilon);
		leftNode.clipTo(rightNode);
		rightNode.clipTo(leftNode);
		rightNode.invert();
		rightNode.clipTo(leftNode);
		rightNode.invert();
		leftNode.build(rightNode.allPolygons());
		return new BSPSolid(leftNode.allPolygons(), this._epsilon);
	}

	public subtract(other: BSPSolid): BSPSolid {
		const leftNode = new BSPNode(this.clone().polygons, this._epsilon);
		const rightNode = new BSPNode(other.clone().polygons, this._epsilon);
		leftNode.invert();
		leftNode.clipTo(rightNode);
		rightNode.clipTo(leftNode);
		rightNode.invert();
		rightNode.clipTo(leftNode);
		rightNode.invert();
		leftNode.build(rightNode.allPolygons());
		leftNode.invert();
		return new BSPSolid(leftNode.allPolygons(), this._epsilon);
	}

	public intersect(other: BSPSolid): BSPSolid {
		const leftNode = new BSPNode(this.clone().polygons, this._epsilon);
		const rightNode = new BSPNode(other.clone().polygons, this._epsilon);
		leftNode.invert();
		rightNode.clipTo(leftNode);
		rightNode.invert();
		leftNode.clipTo(rightNode);
		rightNode.clipTo(leftNode);
		leftNode.build(rightNode.allPolygons());
		leftNode.invert();
		return new BSPSolid(leftNode.allPolygons(), this._epsilon);
	}

	public xor(other: BSPSolid): BSPSolid {
		const leftDiff = this.subtract(other);
		const rightDiff = other.subtract(this);
		return leftDiff.union(rightDiff);
	}
}

const POLYGON_COPLANAR = 0;
const POLYGON_FRONT = 1;
const POLYGON_BACK = 2;
const POLYGON_SPANNING = POLYGON_FRONT | POLYGON_BACK;

const _builtinSolver: CSGSolverAdapter = {
	id: "builtin",
	solve(request: CSGSolveRequest): CSGSolveResult {
		const context: SolveContext = {
			options: request.options,
			diagnostics: [],
			attributeDropWarningIssued: false,
			operandStateBySource: new Map(),
		};
		const solid = evaluateGraphToSolid(request.graph, context);
		const triangleCount = countSolidTriangles(solid);
		if (triangleCount > request.options.maxOutputTriangles) {
			throw new CSGBuildError({
				code: "csg-output-triangle-limit",
				message:
					`CSG output triangles (${triangleCount}) exceed limit ` +
					`(${request.options.maxOutputTriangles})`,
				severity: "error",
			});
		}

		const mesh = solidToMeshAsset(
			solid,
			request.options.capMaterial ?? resolveGraphFallbackMaterial(request.graph)
		);
		return {
			meshAsset: mesh,
			triangleCount,
			diagnostics: context.diagnostics,
		};
	},
};

export function registerWasmCSGSolver(adapter: CSGSolverAdapter): void {
	if (!adapter.id || adapter.id.trim().length === 0) {
		throw new Error("CSG wasm solver id must be a non-empty string");
	}
	_registeredWasmSolvers.set(adapter.id, adapter);
}

export function unregisterWasmCSGSolver(id: string): void {
	_registeredWasmSolvers.delete(id);
}

export function listWasmCSGSolvers(): string[] {
	return Array.from(_registeredWasmSolvers.keys());
}

export function buildCSGMeshAsset(
	graph: CSGGraph,
	options: CSGBuildOptions = {}
): CSGRebuildResult {
	const diagnostics: CSGDiagnostic[] = [];
	const resolved = resolveBuildOptions(graph, options);
	const selected = selectSolver(resolved.solverPreference);
	let fallbackUsed = false;
	let solverId = selected.id;

	try {
		const result = selected.solve({
			graph,
			options: resolved,
		});
		if (result.diagnostics?.length) {
			diagnostics.push(...result.diagnostics);
		}
		return {
			ok: true,
			meshAsset: result.meshAsset,
			triangleCount: result.triangleCount,
			solverId,
			fallbackUsed,
			stale: false,
			diagnostics,
		};
	} catch (error) {
		if (
			resolved.solverPreference === "auto" &&
			selected.id !== _builtinSolver.id
		) {
			fallbackUsed = true;
			diagnostics.push({
				code: "csg-solver-auto-fallback",
				message:
					`CSG solver "${selected.id}" failed and fell back to ` +
					`"${_builtinSolver.id}"`,
				severity: "warning",
				context: {
					cause: error instanceof Error ? error.message : String(error),
				},
			});
			solverId = _builtinSolver.id;
			try {
				const fallback = _builtinSolver.solve({
					graph,
					options: resolved,
				});
				if (fallback.diagnostics?.length) {
					diagnostics.push(...fallback.diagnostics);
				}
				return {
					ok: true,
					meshAsset: fallback.meshAsset,
					triangleCount: fallback.triangleCount,
					solverId,
					fallbackUsed,
					stale: false,
					diagnostics,
				};
			} catch (fallbackError) {
				appendFailureDiagnostic(diagnostics, fallbackError);
			}
		} else {
			appendFailureDiagnostic(diagnostics, error);
		}
	}

	return {
		ok: false,
		meshAsset: null,
		triangleCount: 0,
		solverId,
		fallbackUsed,
		stale: false,
		diagnostics,
	};
}

function appendFailureDiagnostic(
	diagnostics: CSGDiagnostic[],
	error: unknown
): void {
	if (error instanceof CSGBuildError) {
		diagnostics.push(error.diagnostic);
		return;
	}
	diagnostics.push({
		code: "csg-build-failed",
		message:
			error instanceof Error ? error.message : "CSG build failed unexpectedly",
		severity: "error",
	});
}

function selectSolver(preference: CSGSolverPreference): CSGSolverAdapter {
	if (preference === "builtin") {
		return _builtinSolver;
	}
	if (preference === "wasm") {
		const first = _registeredWasmSolvers.values().next().value;
		if (!first) {
			throw new CSGBuildError({
				code: "csg-solver-missing",
				message:
					'CSG solver preference is "wasm" but no wasm solver is registered',
				severity: "error",
			});
		}
		return first;
	}
	const first = _registeredWasmSolvers.values().next().value;
	return first ?? _builtinSolver;
}

function resolveBuildOptions(
	graph: CSGGraph,
	options: CSGBuildOptions
): CSGResolvedBuildOptions {
	return {
		epsilon:
			Number.isFinite(options.epsilon) && (options.epsilon ?? 0) > 0 ?
				Math.max(1e-8, Number(options.epsilon))
			:	DEFAULT_EPSILON,
		maxOutputTriangles:
			Number.isFinite(options.maxOutputTriangles) &&
			(options.maxOutputTriangles ?? 0) > 0 ?
				Math.floor(options.maxOutputTriangles as number)
			:	DEFAULT_MAX_OUTPUT_TRIANGLES,
		capMaterial:
			options.capMaterial ?? resolveGraphFallbackMaterial(graph),
		solverPreference: options.solverPreference ?? "auto",
	};
}

function resolveGraphFallbackMaterial(graph: CSGGraph): Material {
	const leftmost = resolveLeftmostLeaf(graph);
	const operand = resolveOperandDescriptor(leftmost.operand);
	const firstMaterial = operand.mesh.primitives[0]?.material ?? null;
	return operand.material ?? firstMaterial ?? new Material({ name: "CSGCap" });
}

function resolveLeftmostLeaf(graph: CSGGraph): CSGLeafNode {
	if (graph.kind === "leaf") return graph;
	return resolveLeftmostLeaf(graph.left);
}

function evaluateGraphToSolid(
	graph: CSGGraph,
	context: SolveContext
): BSPSolid {
	if (graph.kind === "leaf") {
		const polygons = operandToPolygons(graph.operand, context);
		return new BSPSolid(polygons, context.options.epsilon);
	}

	const left = evaluateGraphToSolid(graph.left, context);
	const right = evaluateGraphToSolid(graph.right, context);
	switch (graph.operation) {
		case "union":
			return left.union(right);
		case "subtract":
			return left.subtract(right);
		case "intersect":
			return left.intersect(right);
		case "xor":
			return left.xor(right);
		default:
			throw new CSGBuildError({
				code: "csg-operation-unknown",
				message: `Unsupported CSG operation "${String(graph.operation)}"`,
				severity: "error",
			});
	}
}

function operandToPolygons(
	operand: CSGOperand,
	context: SolveContext
): BSPPolygon[] {
	const descriptor = resolveOperandDescriptor(operand);
	const epsilon = context.options.epsilon;
	const transform = descriptor.transform;
	const sourceKey =
		descriptor.label ??
		descriptor.mesh.id +
			(transform ? `:${matrixSignature(transform)}` : "");
	const sourceState =
		context.operandStateBySource.get(sourceKey) ?? {
			materials: [],
		};
	context.operandStateBySource.set(sourceKey, sourceState);

	const edgeCounts = new Map<string, number>();
	const polygons: BSPPolygon[] = [];

	for (const primitive of descriptor.mesh.primitives) {
		const topology = primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		if (topology !== DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) {
			throw new CSGBuildError({
				code: "csg-input-non-triangle-topology",
				message:
					`CSG requires triangle-list primitives; got "${topology}" ` +
					`from primitive "${primitive.id}"`,
				severity: "error",
				context: {
					source: sourceKey,
					primitiveId: primitive.id,
				},
			});
		}
		const positions = primitive.geometry.positions;
		const normals = primitive.geometry.normals;
		const uv0 = primitive.geometry.uv0;
		const indices = primitive.geometry.indices;
		if (indices.length % 3 !== 0) {
			throw new CSGBuildError({
				code: "csg-input-invalid-indices",
				message:
					`CSG primitive "${primitive.id}" has invalid index count ` +
					`(${indices.length}), expected triangle triplets`,
				severity: "error",
				context: {
					source: sourceKey,
					primitiveId: primitive.id,
				},
			});
		}
		const vertexCount = Math.floor(positions.length / 3);
		for (let triangleIndex = 0; triangleIndex < indices.length; triangleIndex += 3) {
			const i0 = indices[triangleIndex];
			const i1 = indices[triangleIndex + 1];
			const i2 = indices[triangleIndex + 2];
			if (
				i0 >= vertexCount ||
				i1 >= vertexCount ||
				i2 >= vertexCount
			) {
				throw new CSGBuildError({
					code: "csg-input-index-out-of-range",
					message:
						`CSG primitive "${primitive.id}" contains out-of-range index`,
					severity: "error",
					context: {
						source: sourceKey,
						primitiveId: primitive.id,
						vertexCount,
						indexes: [i0, i1, i2],
					},
				});
			}

			const v0 = buildVertexPayload(
				i0,
				positions,
				normals,
				uv0,
				transform
			);
			const v1 = buildVertexPayload(
				i1,
				positions,
				normals,
				uv0,
				transform
			);
			const v2 = buildVertexPayload(
				i2,
				positions,
				normals,
				uv0,
				transform
			);

			if (
				!context.attributeDropWarningIssued &&
				hasDroppedAttributes(primitive.geometry)
			) {
				context.attributeDropWarningIssued = true;
				context.diagnostics.push({
					code: "csg-attribute-dropped",
					message:
						"CSG v1 only preserves positions/normals/uv0; other attributes were dropped",
					severity: "warning",
					context: {
						source: sourceKey,
						primitiveId: primitive.id,
					},
				});
			}

			const shared: PolygonShared = {
				material: descriptor.material ?? primitive.material,
				source: sourceKey,
				generated: false,
			};
			polygons.push(
				new BSPPolygon(
					[new BSPVertex(v0), new BSPVertex(v1), new BSPVertex(v2)],
					shared,
					epsilon
				)
			);
			sourceState.materials.push(shared.material);
			addTriangleEdges(v0.position, v1.position, v2.position, epsilon, edgeCounts);
		}
	}

	validateManifold(edgeCounts, sourceKey);
	return polygons;
}

function hasDroppedAttributes(
	geometry: NonNullable<MeshAsset["primitives"][number]["geometry"]>
): boolean {
	return (
		(geometry.uv1?.length ?? 0) > 0 ||
		(geometry.colors?.length ?? 0) > 0 ||
		(geometry.joints0?.length ?? 0) > 0 ||
		(geometry.weights0?.length ?? 0) > 0 ||
		(geometry.joints1?.length ?? 0) > 0 ||
		(geometry.weights1?.length ?? 0) > 0 ||
		(geometry.morphTargets?.length ?? 0) > 0
	);
}

function validateManifold(
	edgeCounts: Map<string, number>,
	source: string
): void {
	for (const [edgeKey, count] of edgeCounts.entries()) {
		if (count === 2) continue;
		throw new CSGBuildError({
			code: "csg-input-non-manifold",
			message:
				`CSG input "${source}" is not a closed manifold (edge "${edgeKey}" has ${count} incident faces)`,
			severity: "error",
			context: {
				source,
				edgeKey,
				incidentFaces: count,
			},
		});
	}
}

function buildVertexPayload(
	index: number,
	positions: Float32Array,
	normals: Float32Array | null | undefined,
	uv0: Float32Array | null | undefined,
	transform: Matrix4 | null
): VertexPayload {
	const positionOffset = index * 3;
	const uvOffset = index * 2;
	const position = new Vector3(
		positions[positionOffset],
		positions[positionOffset + 1],
		positions[positionOffset + 2]
	);
	const rawNormal =
		normals ?
			new Vector3(
				normals[positionOffset],
				normals[positionOffset + 1],
				normals[positionOffset + 2]
			)
		:	new Vector3(0, 1, 0);
	const transformedPosition =
		transform ? Matrix4.transformPoint(transform, position) : position;
	const transformedNormal =
		transform ? Matrix4.transformDirection(transform, rawNormal) : rawNormal;

	return {
		position: new Vector3(
			transformedPosition.x,
			transformedPosition.y,
			transformedPosition.z
		),
		normal: normalizeVector(transformedNormal),
		u: uv0?.[uvOffset] ?? 0,
		v: uv0?.[uvOffset + 1] ?? 0,
	};
}

function solidToMeshAsset(solid: BSPSolid, capMaterial: Material): MeshAsset {
	const faces: MeshFace[] = [];
	for (const polygon of solid.polygons) {
		if (polygon.vertices.length < 3) continue;
		for (let index = 1; index < polygon.vertices.length - 1; index++) {
			const a = polygon.vertices[0];
			const b = polygon.vertices[index];
			const c = polygon.vertices[index + 1];
			const material = polygon.shared.generated ? capMaterial : polygon.shared.material;
			faces.push({
				material,
				vertices: [a, b, c].map((vertex) => ({
					x: vertex.position.x,
					y: vertex.position.y,
					z: vertex.position.z,
					u: vertex.u,
					v: vertex.v,
					normal: {
						x: vertex.normal.x,
						y: vertex.normal.y,
						z: vertex.normal.z,
					},
				})),
				normal: {
					x: polygon.plane.normal.x,
					y: polygon.plane.normal.y,
					z: polygon.plane.normal.z,
				},
			});
		}
	}

	return MeshAsset.fromFaces(faces);
}

function countSolidTriangles(solid: BSPSolid): number {
	let count = 0;
	for (const polygon of solid.polygons) {
		count += Math.max(0, polygon.vertices.length - 2);
	}
	return count;
}

function resolveOperandDescriptor(operand: CSGOperand): {
	mesh: MeshAsset;
	transform: Matrix4 | null;
	material: Material | null;
	label?: string;
} {
	if (operand instanceof MeshAsset) {
		return {
			mesh: operand,
			transform: null,
			material: null,
			label: operand.id,
		};
	}
	if (operand instanceof MeshInstance) {
		return {
			mesh: operand.mesh,
			transform: operand.worldMatrix.clone(),
			material: null,
			label: operand.id,
		};
	}
	const descriptor = operand as CSGOperandDescriptor;
	const mesh = descriptor.mesh instanceof MeshInstance ? descriptor.mesh.mesh : descriptor.mesh;
	const transform =
		descriptor.transform ??
		(descriptor.mesh instanceof MeshInstance ?
			descriptor.mesh.worldMatrix.clone()
		:	null);
	return {
		mesh,
		transform: transform ? transform.clone() : null,
		material: descriptor.material ?? null,
		label: descriptor.label,
	};
}

function matrixSignature(matrix: Matrix4): string {
	const elements = matrix.elements;
	let hash = 2166136261;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			hash = Math.imul(hash ^ floatToBits(elements[row][col]), 16777619);
		}
	}
	return (hash >>> 0).toString(16);
}

function floatToBits(value: number): number {
	if (!Number.isFinite(value)) return 0;
	const scratch = new DataView(new ArrayBuffer(4));
	scratch.setFloat32(0, value, true);
	return scratch.getUint32(0, true);
}

function addTriangleEdges(
	a: Vector3,
	b: Vector3,
	c: Vector3,
	epsilon: number,
	edgeCounts: Map<string, number>
): void {
	accumulateEdge(edgeCounts, edgeKey(a, b, epsilon));
	accumulateEdge(edgeCounts, edgeKey(b, c, epsilon));
	accumulateEdge(edgeCounts, edgeKey(c, a, epsilon));
}

function accumulateEdge(map: Map<string, number>, key: string): void {
	map.set(key, (map.get(key) ?? 0) + 1);
}

function edgeKey(a: Vector3, b: Vector3, epsilon: number): string {
	const left = quantizedPointKey(a, epsilon);
	const right = quantizedPointKey(b, epsilon);
	return left < right ? `${left}|${right}` : `${right}|${left}`;
}

function quantizedPointKey(point: Vector3, epsilon: number): string {
	return [
		Math.round(point.x / epsilon),
		Math.round(point.y / epsilon),
		Math.round(point.z / epsilon),
	].join(":");
}

function addVector(a: Vector3, b: Vector3): Vector3 {
	return Vector3.add(a, b);
}

function subVector(a: Vector3, b: Vector3): Vector3 {
	return Vector3.sub(a, b);
}

function scaleVector(vector: Vector3, scalar: number): Vector3 {
	return Vector3.scale(vector, scalar);
}

function dotVector(a: Vector3, b: Vector3): number {
	return Vector3.dot(a, b);
}

function crossVector(a: Vector3, b: Vector3): Vector3 {
	return Vector3.cross(a, b);
}

function normalizeVector(vector: Vector3): Vector3 {
	const length = Vector3.length(vector);
	if (length <= DEFAULT_EPSILON) {
		return new Vector3(0, 1, 0);
	}
	return Vector3.scale(vector, 1 / length);
}

function clamp01(value: number): number {
	if (!Number.isFinite(value)) return 0;
	if (value <= 0) return 0;
	if (value >= 1) return 1;
	return value;
}

export function createEmptyCSGResult(
	solverId: string = _builtinSolver.id
): CSGRebuildResult {
	return {
		ok: false,
		meshAsset: null,
		triangleCount: 0,
		solverId,
		fallbackUsed: false,
		stale: false,
		diagnostics: [],
	};
}

export function expandBoundsForDiagnostics(mesh: MeshAsset): {
	min: Vector3;
	max: Vector3;
} {
	return {
		min: new Vector3(
			mesh.boundingBox.min.x,
			mesh.boundingBox.min.y,
			mesh.boundingBox.min.z
		),
		max: new Vector3(
			mesh.boundingBox.max.x,
			mesh.boundingBox.max.y,
			mesh.boundingBox.max.z
		),
	};
}

export function resolveCSGBoundingGuard(mesh: MeshAsset): boolean {
	return (
		Math.abs(mesh.boundingBox.min.x) < LARGE_VALUE &&
		Math.abs(mesh.boundingBox.min.y) < LARGE_VALUE &&
		Math.abs(mesh.boundingBox.min.z) < LARGE_VALUE &&
		Math.abs(mesh.boundingBox.max.x) < LARGE_VALUE &&
		Math.abs(mesh.boundingBox.max.y) < LARGE_VALUE &&
		Math.abs(mesh.boundingBox.max.z) < LARGE_VALUE
	);
}
