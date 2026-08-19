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
	},
	projects: [
		{
			name: "webgl",
			testMatch: "**/webgl_*.spec.ts",
			use: {
				launchOptions: {
					args: ["--use-gl=angle", "--use-angle=swiftshader"],
				},
			},
		},
		{
			name: "webgpu",
			testMatch: "**/webgpu_*.spec.ts",
			use: {
				launchOptions: {
					args: [
						"--enable-gpu",
						"--enable-unsafe-webgpu",
						"--use-webgpu-adapter=swiftshader",
						"--use-vulkan=swiftshader",
						"--enable-features=Vulkan",
						"--enable-dawn-features=allow_unsafe_apis",
						"--disable-dawn-features=use_dxc",
						"--use-gpu-in-tests",
					],
				},
			},
		},
	],
	webServer: {
		command: `bun x vite --host ${host} --port ${port}`,
		port,
		reuseExistingServer: true,
		timeout: 120_000,
	},
});
