import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
	renderGraphNodeId,
	renderGraphPhysicalResourceId,
	renderGraphResourceId,
} from "../../../src/rendergraph/types.ts";

const fixture = fileURLToPath(new URL(
	"./fixtures/render_graph_id_types.ts",
	import.meta.url,
));
const compiler = fileURLToPath(new URL(
	"../../../node_modules/typescript/bin/tsc",
	import.meta.url,
));
const result = spawnSync(process.execPath, [
	compiler,
	"--allowImportingTsExtensions",
	"--module", "ESNext",
	"--moduleResolution", "bundler",
	"--noEmit",
	"--skipLibCheck",
	"--target", "ESNext",
	"--types", "@webgpu/types,vite/client",
	fixture,
], {
	encoding: "utf8",
});
assert.equal(
	result.status,
	0,
	[result.stdout, result.stderr].filter(Boolean).join("\n"),
);

assert.equal(renderGraphResourceId("resource"), "resource");
assert.equal(renderGraphNodeId("node"), "node");
assert.equal(renderGraphPhysicalResourceId("physical"), "physical");

console.log("Render graph branded ID type tests passed");
