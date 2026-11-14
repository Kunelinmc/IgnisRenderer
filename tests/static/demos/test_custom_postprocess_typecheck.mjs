import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const compiler = fileURLToPath(
	new URL("../../../node_modules/typescript/bin/tsc", import.meta.url),
);

execFileSync(process.execPath, [
	compiler,
	"--ignoreConfig",
	"--strict", "false",
	"--noEmit",
	"--target", "ESNext",
	"--module", "ESNext",
	"--moduleResolution", "bundler",
	"--lib", "ESNext,DOM,DOM.Iterable",
	"--skipLibCheck",
	"--allowImportingTsExtensions",
	"demos/custom-postprocess/main.ts",
	"src/vite-env.d.ts",
], { cwd: root, stdio: "inherit" });

console.log("Custom post-process demo typecheck passed");
