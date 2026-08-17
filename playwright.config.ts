import { defineConfig } from "@playwright/test";

const host = "127.0.0.1";
const port = 4173;
const baseURL = `http://${host}:${port}`;

export default defineConfig({
	testDir: "./tests/browser/playwright",
	fullyParallel: false,
	retries: 0,
	workers: 1,
	use: {
		baseURL,
		browserName: "chromium",
		headless: true,
		launchOptions: {
			args: ["--use-gl=angle", "--use-angle=swiftshader"],
		},
	},
	webServer: {
		command: `bun x vite --host ${host} --port ${port}`,
		port,
		reuseExistingServer: true,
		timeout: 120_000,
	},
});
