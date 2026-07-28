# BVH Loader Contract
## Scope
This document defines the contract for `BVHLoader` in `src/loaders/BVHLoader.ts`.
Implementations and callers must follow this contract when loading `.bvh`
motion files into IgnisRenderer.

## Background
`BVHLoader` provides a native path for importing Biovision Hierarchy motion
data into the existing `Node` + animation runtime flow. It should interoperate
with `AnimationClip`, `KeyframeTrack`, and `EntityPrefab` contracts without
introducing a parallel animation runtime.

## API/Contract
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

## Errors & Diagnostics
- Invalid BVH structure (for example missing `HIERARCHY`, `MOTION`, braces,
  or malformed `CHANNELS`) must throw an `Error`.
- Invalid numeric tokens (for example non-finite `OFFSET`, `Frame Time`, or
  motion values) must throw an `Error`.
- Truncated motion payloads (fewer values than `Frames * channelCount`) must
  throw an `Error`.
- `load`/`loadPrefab` network failures must emit `error` and rethrow.

## Compatibility / Breaking Changes
This change adds a new loader export (`BVHLoader`) and does not remove or
modify existing loader APIs. Existing code should remain compatible.
