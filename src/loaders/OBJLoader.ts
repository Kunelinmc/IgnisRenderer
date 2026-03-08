import { Node } from "../core/Node";
import { Loader, type ParseProgressEvent } from "./Loader";
import type { IVector3 } from "../maths/types";
import {
	MeshAsset,
	MeshInstance,
	type MeshFace,
	type MeshVertex,
} from "../meshes";

/**
 * OBJLoader parses .obj files and creates a scene graph root node.
 */
export class OBJLoader extends Loader {
	constructor() {
		super();
	}

	/**
	 * Loads an OBJ file from a URL.
	 */
	public async load(url: string): Promise<Node> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const text = new TextDecoder().decode(buffer);
			const root = this.parse(text);
			this.emit("load", root);
			return root;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}

	/**
	 * Parses OBJ text.
	 */
	public parse(text: string): Node {
		this.emit("parsestart");
		const vertices: IVector3[] = [];
		const uvs: { u: number; v: number }[] = [];
		const normals: IVector3[] = [];
		const faces: MeshFace[] = [];

		const lines = text.split("\n");
		const lineCount = lines.length;

		for (let i = 0; i < lineCount; i++) {
			const line = lines[i].trim();
			if (i % 1000 === 0) {
				this.emit("parseprogress", {
					current: i,
					total: lineCount,
					message: `Parsing line ${i}/${lineCount}`,
				} as ParseProgressEvent);
			}
			if (!line || line.startsWith("#")) continue;

			const parts = line.split(/\s+/);
			const type = parts[0];

			if (type === "v") {
				vertices.push({
					x: parseFloat(parts[1]),
					y: parseFloat(parts[2]),
					z: parseFloat(parts[3]),
				});
			} else if (type === "vt") {
				uvs.push({
					u: parseFloat(parts[1]),
					v: parseFloat(parts[2]),
				});
			} else if (type === "vn") {
				normals.push({
					x: parseFloat(parts[1]),
					y: parseFloat(parts[2]),
					z: parseFloat(parts[3]),
				});
			} else if (type === "f") {
				const faceVertices: MeshVertex[] = [];
				let faceNormal: IVector3 | undefined;

				for (let j = 1; j < parts.length; j++) {
					const indices = parts[j].split("/");
					const vertexIndex = parseInt(indices[0]) - 1;
					const uvIndex = indices[1] ? parseInt(indices[1]) - 1 : -1;
					const normalIndex = indices[2] ? parseInt(indices[2]) - 1 : -1;

					const vertex: MeshVertex = { ...vertices[vertexIndex] };
					if (uvIndex >= 0) {
						vertex.u = uvs[uvIndex].u;
						vertex.v = uvs[uvIndex].v;
					} else {
						vertex.u = 0;
						vertex.v = 0;
					}

					if (normalIndex >= 0) {
						vertex.normal = { ...normals[normalIndex] };
						if (!faceNormal) {
							faceNormal = vertex.normal;
						}
					}

					faceVertices.push(vertex);
				}

				faces.push({
					vertices: faceVertices,
					normal: faceNormal,
				});
			}
		}

		const mesh = MeshAsset.fromFaces(faces);
		const root = new Node({
			idPrefix: "node",
			name: "objRoot",
		});
		root.addChild(
			new MeshInstance({
				mesh,
				name: "objMesh",
			})
		);

		this.emit("parseend", root);
		return root;
	}
}
