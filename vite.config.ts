import { defineConfig } from "vite";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const _dirname = dirname(fileURLToPath(import.meta.url));
const entryPath = process.env.ENTRY || "index.html";

export default defineConfig({
	base: "./",
	build: {
		rollupOptions: {
			input: {
				main: resolve(_dirname, entryPath),
			},
			output: {
				entryFileNames: "assets/[name].js",
				chunkFileNames: "assets/[name].js",
				assetFileNames: "assets/[name].[ext]",
			},
		},
	},
});
