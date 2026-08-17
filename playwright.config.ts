import { defineConfig } from "@playwright/test";

export default defineConfig({
	testDir: "./tests/browser/playwright",
	fullyParallel: false,
	retries: 0,
	workers: 1,
	use: {
		baseURL: "http://127.0.0.1:4173",
		browserName: "chromium",
		headless: true,
		launchOptions: {
			args: ["--use-gl=angle", "--use-angle=swiftshader"],
		},
	},
	webServer: {
		command: "bun x vite --host 127.0.0.1 --port 4173",
		url:
			"http://127.0.0.1:4173/tests/browser/fixtures/webgl_auxiliary_raster.html",
		reuseExistingServer: true,
		timeout: 120_000,
	},
});
