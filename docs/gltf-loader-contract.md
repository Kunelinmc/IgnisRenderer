# glTF Loader Contract

## Scope

This document defines the contract for `GLTFLoader` in
`src/loaders/GLTFLoader.ts`. It specifies how `.gltf` (JSON) and `.glb`
(binary) models are loaded, parsed, and converted to the IgnisRenderer node
hierarchy, ECS prefabs, materials, animations, and cameras.

## Background

`GLTFLoader` is a native loader for glTF 2.0 asset resources. It parses files to construct a `Node` hierarchy with `MeshInstance` components, lights, cameras, skeletal joint skins, animations, and morph targets.

## API/Contract

- `new GLTFLoader()` must construct a loader with event behavior compatible with `Loader`.
- `GLTFLoader.load(url)` must asynchronously fetch the asset at `url`, parse it, emit the `load` event with the root `Node`, and return the root `Node`.
- `GLTFLoader.loadPrefab(url)` must asynchronously load the asset at `url`, parse it into an `EntityPrefab`, emit the `loadprefab` event, and return the `EntityPrefab`.
- `GLTFLoader.parse(data, baseURL?)` must accept an `ArrayBuffer` containing GLB or glTF JSON data, parse it, and return a `Promise` resolving to the root `Node` of the loaded scene graph.
- `GLTFLoader.parsePrefab(data, baseURL?)` must parse an in-memory `ArrayBuffer` of GLB/glTF data and return a `Promise` resolving to an `EntityPrefab` (specifically `NodeEntityPrefab`).
- `GLTFLoader.getLastAnimationBundle()` must return a `GLTFAnimationBundle` containing parsed `AnimationClip` objects, `Skeleton` structures, morph bindings, and a node path-to-ID map, or `null` if no parsing has completed.
- `GLTFLoader.clearLastAnimationBundle()` must clear the cached animation bundle.
- `GLTFLoader.getAccessorData(json, buffers, index)` must retrieve data from the specified accessor `index`, applying data type conversions, normalization, and sparse overrides where defined.

### Supported Features & Extensions

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

### Texture Color Spaces

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

### Primitive Topology & Modes

The loader must parse glTF primitive modes and convert them to IgnisRenderer `PrimitiveDrawTopology` ("point-list", "line-list", or "triangle-list") as follows:
- Mode `0` (`POINTS`) must yield `"point-list"`.
- Mode `1` (`LINES`), `2` (`LINE_LOOP`), and `3` (`LINE_STRIP`) must yield `"line-list"`.
- Mode `4` (`TRIANGLES`), `5` (`TRIANGLE_STRIP`), and `6` (`TRIANGLE_FAN`) must yield `"triangle-list"`.
The loader must normalize or reconstruct indices for loop, strip, and fan topologies.

### Functional Limitations & Security Constraints

- **glTF Version**: The loader must only support glTF 2.0. Any glTF file with a version other than `2` must throw an error.
- **Hierarchical Depth**: Node hierarchies must not exceed a maximum depth of 1024 (`MAX_NODE_DEPTH`). Deeper hierarchies must be rejected to prevent call stack overflow.
- **Cyclic Hierarchies**: Cyclic child-parent node relationships must be detected. If a node index is encountered twice within the active traversal path, an error must be thrown.
- **Resource URI Schemes**: The loader must reject resource URIs using unsupported schemes. Only `http:`, `https:`, and `data:` schemes are allowed. Local file access or parent directory traversal scheme (e.g., `file:///etc/passwd`) must be rejected.
- **Accessor Allocation Limits**: Individual accessors must not exceed a size limit of 256 MB (`MAX_ACCESSOR_BYTE_LENGTH`) to prevent memory exhaustion attacks.
- **Image Decode Concurrency**: Image decoding must be restricted to a maximum concurrency of 2 (`GLTF_IMAGE_DECODE_CONCURRENCY`) to manage memory and performance constraints.
- **Sparse Index Bounds**: Sparse accessor indices must fall within the overall accessor count boundary. Index values outside this range must throw an error.

## Usage

### Loading a glTF Scene Graph

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

### Loading and Instantiating an Entity Prefab

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

### Verification Command

Static loader tests are executed using:

```bash
bun tests/static/loaders/test_gltf_primitive_modes.mjs
bun tests/static/loaders/test_gltf_material_extensions.mjs
bun tests/static/loaders/test_gltf_loader_security.mjs
bun tests/static/loaders/test_gltf_prefab_contract.mjs
```

## Errors & Diagnostics

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

## Compatibility / Breaking Changes

N/A
