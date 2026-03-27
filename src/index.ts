export { Vector2 } from "./maths/Vector2";
export { Vector3 } from "./maths/Vector3";
export { Vector4 } from "./maths/Vector4";
export { Box2 } from "./maths/Box2";
export { Box3 } from "./maths/Box3";
export { Matrix3 } from "./maths/Matrix3";
export { Matrix4 } from "./maths/Matrix4";
export { Quaternion } from "./maths/Quaternion";

export * from "./maths/Common";
export * from "./maths/types";
export * from "./foundation/Color";
export * from "./foundation/Logger";
export * from "./foundation/Platform";
export { Node } from "./core/Node";
export { MeshAsset, MeshInstance, MeshFactory } from "./meshes";
export type {
	IVertex,
	IPrimitive,
	IPrimitiveGeometry,
	BoundingSphere,
	BoundingBox,
} from "./core/types";
export { Renderer } from "./renderers/Renderer";
export { EventEmitter } from "./core/EventEmitter";
export * from "./workers";
export { SoftwareBackend } from "./renderers/SoftwareBackend";
export type {
	SoftwareRasterMode,
	SoftwareTileOptions,
	SoftwareBackendOptions,
} from "./renderers/software/SoftwareRasterConfig";
export { WebGPUBackend } from "./renderers/WebGPUBackend";
export { WebGLBackend } from "./renderers/WebGLBackend";
export type {
	WebGPUPostProcessPassPlugin,
	WebGPUPostProcessPassKind,
} from "./renderers/webgpu/WebGPUPostProcessGraph";
export { Rasterizer } from "./renderers/software/Rasterizer";
export { PostProcessor } from "./renderers/software/PostProcessor";
export type {
	SSROptions,
	SSAOOptions,
	TAAOptions,
	MotionBlurOptions,
	DOFOptions,
	VolumetricOptions,
} from "./pipeline/types";
export { Scene } from "./core/Scene";
export { Camera, CameraType } from "./cameras/Camera";
export { OrthographicCamera } from "./cameras/OrthographicCamera";
export { OrbitCamera } from "./cameras/OrbitCamera";
export { FPSCamera } from "./cameras/FPSCamera";
export {
	CameraShakePlugin,
	type CameraShakePluginOptions,
	type CameraShakeImpulse,
} from "./addons/CameraShakePlugin";
export {
	Material,
	BasicMaterial,
	PhongMaterial,
	GouraudMaterial,
	UnlitMaterial,
	PBRMaterial,
	ShaderMaterial,
} from "./materials";
export * from "./animation";
export * from "./particles";
export * from "./physics";
export * from "./lights";
export { Texture, type TextureColorSpace } from "./core/Texture";
export { VideoTexture, type VideoTextureParams } from "./core/VideoTexture";
export { TextureLoader } from "./loaders/TextureLoader";
export { OBJLoader } from "./loaders/OBJLoader";
export { GLTFLoader } from "./loaders/GLTFLoader";
export { HDRLoader } from "./loaders/HDRLoader";
export { Loader } from "./loaders/Loader";
export * as experimentalECS from "./ecs";
export type { EntityPrefab } from "./ecs";
