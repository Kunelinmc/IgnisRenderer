import {
	SimpleModel,
	type ModelFace,
	type ModelVertex,
} from "../models/SimpleModel";
import { Loader, type ParseProgressEvent } from "./Loader";
import type { IVector3 } from "../maths/types";

/**
 * OBJLoader parses .obj files and creates SimpleModel objects.
 */
export class OBJLoader extends Loader {
	constructor() {
		super();
	}

	/**
	 * Loads an OBJ file from a URL.
	 */
	public async load(url: string): Promise<SimpleModel> {
		try {
			const buffer = await this._fetchWithProgress(url);
			const text = new TextDecoder().decode(buffer);
			const model = this.parse(text);
			this.emit("load", model);
			return model;
		} catch (error) {
			this.emit("error", error);
			throw error;
		}
	}

	/**
	 * Parses OBJ text.
	 */
	public parse(text: string): SimpleModel {
		this.emit("parsestart");
		const vertices: IVector3[] = [];
		const uvs: { u: number; v: number }[] = [];
		const normals: IVector3[] = [];
		const faces: ModelFace[] = [];

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
				const faceVertices: ModelVertex[] = [];
				let faceNormal: IVector3 | undefined = undefined;

				// Each part is v/vt/vn
				for (let j = 1; j < parts.length; j++) {
					const indices = parts[j].split("/");
					const vIdx = parseInt(indices[0]) - 1;
					const vtIdx = indices[1] ? parseInt(indices[1]) - 1 : -1;
					const vnIdx = indices[2] ? parseInt(indices[2]) - 1 : -1;

					const v: ModelVertex = { ...vertices[vIdx] };
					if (vtIdx >= 0) {
						v.u = uvs[vtIdx].u;
						// Some OBJ exporters use 1-v for V
						v.v = uvs[vtIdx].v;
					} else {
						v.u = 0;
						v.v = 0;
					}

					// If vertex normals are provided, we could store them per vertex.
					// But current SimpleModel structure stores one normal per face.
					// Let's store the vertex normal if available for potential Gouraud/Phong.
					if (vnIdx >= 0) {
						v.normal = { ...normals[vnIdx] };
						if (faceNormal === undefined) faceNormal = v.normal;
					}

					faceVertices.push(v);
				}

				faces.push({
					vertices: faceVertices,
					normal: faceNormal,
				});
			}
		}

		const model = new SimpleModel(faces);
		this.emit("parseend", model);
		return model;
	}
}
