import { GLTFLoader } from "./GLTFLoader";
/**
 * GLBLoader is now a wrapper around GLTFLoader for backward compatibility.
 * It primarily handles binary .glb files but can also handle .gltf via inheritance.
 */
export class GLBLoader extends GLTFLoader {
	constructor() {
		super();
		console.warn("GLBLoader is deprecated. Use GLTFLoader instead.");
	}
}
