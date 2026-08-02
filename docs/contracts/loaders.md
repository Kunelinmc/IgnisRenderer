# Loader Contract

This document defines model, image, and motion loading behavior for glTF, EXR, and BVH assets.

## Contract

### glTF loading

- `new GLTFLoader()` must construct a loader with event behavior compatible with `Loader`.
- `GLTFLoader.load(url)` must asynchronously fetch the asset at `url`, parse it, emit the `load` event with the root `Node`, and return the root `Node`.
- `GLTFLoader.loadPrefab(url)` must asynchronously load the asset at `url`, parse it into an `EntityPrefab`, emit the `loadprefab` event, and return the `EntityPrefab`.
- `GLTFLoader.parse(data, baseURL?)` must accept an `ArrayBuffer` containing GLB or glTF JSON data, parse it, and return a `Promise` resolving to the root `Node` of the loaded scene graph.
- `GLTFLoader.parsePrefab(data, baseURL?)` must parse an in-memory `ArrayBuffer` of GLB/glTF data and return a `Promise` resolving to an `EntityPrefab` (specifically `NodeEntityPrefab`).
- `GLTFLoader.getLastAnimationBundle()` must return a `GLTFAnimationBundle` containing parsed `AnimationClip` objects, `Skeleton` structures, morph bindings, and a node path-to-ID map, or `null` if no parsing has completed.
- `GLTFLoader.clearLastAnimationBundle()` must clear the cached animation bundle.
- `GLTFLoader.getAccessorData(json, buffers, index)` must retrieve data from the specified accessor `index`, applying data type conversions, normalization, and sparse overrides where defined.

#### Supported Features & Extensions

The loader must support the following glTF 2.0 extensions:
- `KHR_materials_unlit`: Must map materials to `UnlitMaterial`.
- `KHR_materials_emissive_strength`: Must scale emissive factors using the `emissiveStrength` multiplier.
- `KHR_materials_ior`: Must set the index of refraction (`ior`) and update the material's `reflectance`.
- `KHR_materials_specular`: Must parse and assign `specularFactor`, `specularColorFactor`, `specularTexture`, and `specularColorTexture`.
- `KHR_materials_clearcoat`: Must parse and assign clearcoat factors, roughness, normal maps, and scale.
- `KHR_materials_sheen`: Must parse and assign sheen color factors, roughness, and their maps.
- `KHR_materials_iridescence`: Must parse and assign iridescence factor, IOR, thickness range, maps, and calculate iridescence properties in the evaluator.
- `KHR_materials_anisotropy`: Must parse and assign anisotropy strength, rotation, maps, and calculate tangent-space rotation in the evaluator.
- `KHR_materials_transmission`: Must parse and assign transmission factors and maps.
- `KHR_materials_volume`: Must parse and assign thickness factors, attenuation distance, maps, and attenuation color.
- `KHR_texture_transform`: Must support texture offset, scale, rotation, and UV coordinates mapping.
- `KHR_lights_punctual`: Must support directional, point, and spot lights, mapping them to `DirectionalLight`, `PointLight`, `SpotLight`, or `AmbientLight`.

#### Texture Color Spaces

The loader must construct and clone textures with appropriate color spaces:
- `sRGB` must be used for:
  - Albedo maps (`baseColorTexture`)
  - Emissive maps (`emissiveTexture`)
  - Sheen color maps (`sheenColorTexture`)
  - Specular color maps (`specularColorTexture`)
- `Linear` must be used for:
  - Metallic-roughness maps (`metallicRoughnessTexture`)
  - Normal maps (`normalTexture`)
  - Occlusion maps (`occlusionTexture`)
  - Specular maps (`specularTexture`)
  - Sheen roughness maps (`sheenRoughnessTexture`)
  - Transmission maps (`transmissionTexture`)
  - Iridescence maps (`iridescenceTexture`)
  - Iridescence thickness maps (`iridescenceThicknessTexture`)
  - Anisotropy maps (`anisotropyTexture`)
  - Clearcoat maps (`clearcoatTexture` / `clearcoatRoughnessTexture` / `clearcoatNormalTexture`)
  - Thickness maps (`thicknessTexture`)

#### Primitive Topology & Modes

The loader must parse glTF primitive modes and convert them to IgnisRenderer `PrimitiveDrawTopology` ("point-list", "line-list", or "triangle-list") as follows:
- Mode `0` (`POINTS`) must yield `"point-list"`.
- Mode `1` (`LINES`), `2` (`LINE_LOOP`), and `3` (`LINE_STRIP`) must yield `"line-list"`.
- Mode `4` (`TRIANGLES`), `5` (`TRIANGLE_STRIP`), and `6` (`TRIANGLE_FAN`) must yield `"triangle-list"`.
The loader must normalize or reconstruct indices for loop, strip, and fan topologies.

#### Functional Limitations & Security Constraints

- **glTF Version**: The loader must only support glTF 2.0. Any glTF file with a version other than `2` must throw an error.
- **Hierarchical Depth**: Node hierarchies must not exceed a maximum depth of 1024 (`MAX_NODE_DEPTH`). Deeper hierarchies must be rejected to prevent call stack overflow.
- **Cyclic Hierarchies**: Cyclic child-parent node relationships must be detected. If a node index is encountered twice within the active traversal path, an error must be thrown.
- **Resource URI Schemes**: The loader must reject resource URIs using unsupported schemes. Only `http:`, `https:`, and `data:` schemes are allowed. Local file access or parent directory traversal scheme (e.g., `file:///etc/passwd`) must be rejected.
- **Accessor Allocation Limits**: Individual accessors must not exceed a size limit of 256 MB (`MAX_ACCESSOR_BYTE_LENGTH`) to prevent memory exhaustion attacks.
- **Image Decode Concurrency**: Image decoding must be restricted to a maximum concurrency of 2 (`GLTF_IMAGE_DECODE_CONCURRENCY`) to manage memory and performance constraints.
- **Sparse Index Bounds**: Sparse accessor indices must fall within the overall accessor count boundary. Index values outside this range must throw an error.

### EXR loading

- `new EXRLoader()` must create a loader with event behavior compatible with
  `Loader`.
- `EXRLoader.load(url, options?)` must fetch OpenEXR bytes from `url`, parse
  them with `parseAsync`, emit `load` on success, and return a `Texture` with
  `colorSpace === "HDR"`.
- `EXRLoader.load(url, options?)` must return a 1x1 black HDR fallback texture
  marked with `isLoadErrorFallback` when fetching or parsing fails.
- `EXRLoader.parse(buffer, options?)` must synchronously parse single-part
  scanline EXR files that use `NO_COMPRESSION`, `RLE_COMPRESSION`, or
  `PIZ_COMPRESSION`.
- `EXRLoader.parseAsync(buffer, options?)` must parse single-part scanline EXR
  files that use `NO_COMPRESSION`, `RLE_COMPRESSION`, `ZIPS_COMPRESSION`, or
  `ZIP_COMPRESSION`, or `PIZ_COMPRESSION`.
- `ZIPS_COMPRESSION` and `ZIP_COMPRESSION` must require runtime
  `DecompressionStream` support.
- The parser must reject multi-part, tiled, deep, PXR24, B44, B44A, DWAA, DWAB,
  and HTJ2K files with explicit diagnostics.
- The parser must require `R`, `G`, and `B` channels. It may use exact channel
  names or layer-suffixed names such as `beauty.R`.
- The parser must accept `HALF`, `FLOAT`, and `UINT` channel sample types with
  `xSampling === 1` and `ySampling === 1`.
- `EXRLoader.applyToEnvironment(target, texture, options?)` must assign
  `texture` to `environment.backgroundTexture` and `environment.iblTexture` by
  default.
- `EXRLoader.loadEnvironment(url, target, options?)` must load the texture and
  apply it to the provided `Environment` or scene-like target.
- `options.defaultAlpha` must define the alpha value used when the EXR has no
  `A` channel. The default value must be `1`.
- `options.background === false` must skip assignment to
  `environment.backgroundTexture`.
- `options.ibl === false` must skip assignment to `environment.iblTexture`.

### BVH loading

- `new BVHLoader()` must create a loader with event behavior compatible with
  `Loader`.
- `BVHLoader.load(url, options?)` must fetch text from `url`, parse BVH
  hierarchy and motion, and resolve to a `Node` root.
- `BVHLoader.parse(data, options?)` must accept either `string` or
  `ArrayBuffer` BVH input and return a `Node` root.
- `BVHLoader.loadPrefab(url, options?)` must return an `EntityPrefab` that
  contains a parsed root and animation bundle.
- `BVHLoader.parsePrefab(data, options?)` must return an `EntityPrefab` from
  in-memory BVH input.
- `BVHLoader.getLastAnimationBundle()` must return the most recently parsed
  animation bundle or `null` when no parse has completed.
- `BVHLoader.clearLastAnimationBundle()` must clear the cached bundle state.
- `BVHParseOptions.rootName` may override the container root `Node` name.
- `BVHParseOptions.clipName` may override the generated `AnimationClip` name.
- Rotation channels must be applied in source channel order and emitted as
  quaternion `rotation` tracks.
- Position channels must be emitted as `translation` tracks in seconds-based
  keyframe time, and offsets must be preserved as the baseline transform.

## Usage

### glTF loading

#### Loading a glTF Scene Graph

```ts
import { GLTFLoader } from "../src/loaders/GLTFLoader";

const loader = new GLTFLoader();

// Load a binary GLB or JSON glTF model
const rootNode = await loader.load("assets/models/character.glb");

// Access the cached animation bundle parsed during load
const animationBundle = loader.getLastAnimationBundle();
if (animationBundle) {
	console.log(`Loaded ${animationBundle.clips.length} animation clips`);
}
```

#### Loading and Instantiating an Entity Prefab

```ts
import { GLTFLoader } from "../src/loaders/GLTFLoader";
import { Scene } from "../src/core/Scene";

const scene = new Scene();
const loader = new GLTFLoader();

// Load as an EntityPrefab for ECS spawning
const prefab = await loader.loadPrefab("assets/models/character.glb");

// Instantiate the prefab in the scene
const { root, rootEntity } = prefab.instantiate(scene);
```

#### Verification Command

Static loader tests are executed using:

```bash
bun tests/static/loaders/test_gltf_primitive_modes.mjs
bun tests/static/loaders/test_gltf_material_extensions.mjs
bun tests/static/loaders/test_gltf_loader_security.mjs
bun tests/static/loaders/test_gltf_prefab_contract.mjs
```

### EXR loading

```ts
import { EXRLoader, Scene } from "../src";

const scene = new Scene();
const loader = new EXRLoader();

await loader.loadEnvironment("assets/studio.exr", scene.environment);
```

```ts
import { EXRLoader, Scene } from "../src";

const scene = new Scene();
const loader = new EXRLoader();
const texture = await loader.load("assets/studio.exr");

loader.applyToEnvironment(scene, texture, {
	background: true,
	ibl: true,
});
```

### BVH loading

```ts
import { BVHLoader } from "../src/loaders/BVHLoader";

const loader = new BVHLoader();
const root = await loader.load("/motions/walk.bvh", {
	clipName: "walk",
});

const bundle = loader.getLastAnimationBundle();
if (!bundle) {
	throw new Error("Expected animation bundle");
}

const clip = bundle.clips[0];
console.log(root.name, clip.name, clip.duration);
```

```bash
bun tests/static/loaders/test_bvh_loader.mjs
```

## Diagnostics

### glTF loading

- `Invalid GLB: file is too short for header`: Thrown when a binary GLB file is less than 12 bytes.
- `Unsupported GLB version: <version>`: Thrown when the version field in the GLB header is not `2`.
- `Invalid GLB length: declared <len>, actual <actual>`: Thrown when the declared file length in the GLB header exceeds the actual data size.
- `Invalid GLB: incomplete chunk header`: Thrown when a chunk header is truncated or out of bounds.
- `Invalid GLB: chunk exceeds declared file length`: Thrown when chunk boundary goes beyond the header length.
- `Invalid GLB: multiple JSON chunks are not allowed`: Thrown when more than one JSON chunk is present in a GLB file.
- `Failed to parse glTF JSON`: Thrown when the JSON content cannot be parsed.
- `glTF buffer <idx> is missing uri and has no embedded BIN chunk`: Thrown when a `.gltf` references an external buffer without a URI or a GLB lacks a binary chunk.
- `Invalid glTF resource URI: empty value`: Thrown when a buffer or image URI is empty.
- `Unsupported URI scheme <scheme> in glTF resource`: Thrown when trying to load from an unauthorized URI scheme.
- `glTF node hierarchy exceeds safe depth limit (1024)`: Thrown when the traversal depth exceeds `MAX_NODE_DEPTH`.
- `Detected cyclic node hierarchy at node index <idx>`: Thrown when a node hierarchy contains a cycle.
- `Accessor <label> exceeds safe allocation limit (256MB)`: Thrown when an accessor's byte size exceeds `MAX_ACCESSOR_BYTE_LENGTH`.
- `Sparse accessor index <idx> out of bounds for accessor <index>`: Thrown when sparse index value is negative or greater than or equal to the accessor count.
- `Unsupported glTF primitive mode: <mode>`: Thrown when the draw mode is outside `[0, 6]`.

### EXR loading

- `Invalid OpenEXR magic number` must be thrown when `buffer` does not start
  with the OpenEXR magic number.
- `Multi-part EXR files are not supported` must be thrown when the version field
  has the multi-part flag.
- `Deep EXR files are not supported` must be thrown when the version field has
  the deep-data flag.
- `Tiled EXR files are not supported` must be thrown when the version field has
  the tiled flag.
- `Unsupported EXR compression method` must be thrown for compression methods
  outside the supported contract.
- `ZIP/ZIPS compressed EXR data requires parseAsync() or load()` must be thrown
  when `parse` receives a compressed ZIP/ZIPS chunk.
- `ZIP/ZIPS EXR compression requires DecompressionStream support` must be thrown
  when `parseAsync` receives ZIP/ZIPS data in a runtime without
  `DecompressionStream`.
- `EXR image must provide R, G, and B channels` must be thrown when required
  color channels are absent.

### BVH loading

- Invalid BVH structure (for example missing `HIERARCHY`, `MOTION`, braces,
  or malformed `CHANNELS`) must throw an `Error`.
- Invalid numeric tokens (for example non-finite `OFFSET`, `Frame Time`, or
  motion values) must throw an `Error`.
- Truncated motion payloads (fewer values than `Frames * channelCount`) must
  throw an `Error`.
- `load`/`loadPrefab` network failures must emit `error` and rethrow.

## Compatibility

### glTF loading

N/A

### EXR loading

This change adds a new loader export (`EXRLoader`) and does not remove or change
existing loader APIs.

### BVH loading

This change adds a new loader export (`BVHLoader`) and does not remove or
modify existing loader APIs. Existing code should remain compatible.

## Verification

```bash
bun run test:sparse
bunx tsc --noEmit
```

## Related Documents

- [Public renderer guide](../public/renderer.md)
- [Geometry contract](geometry.md)
- [Materials contract](materials.md)
