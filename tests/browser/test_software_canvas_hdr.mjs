// This probe is intentionally browser-native. The repository browser runner may
// execute it in a host without DOM injection, in which case it reports a skip.
if (typeof document === "undefined" || typeof ImageData === "undefined") {
	console.log("SKIP Software Canvas HDR: browser DOM is unavailable");
} else if (!/Chrom(?:e|ium)/.test(navigator.userAgent)) {
	console.log("SKIP Software Canvas HDR: Chromium is the validation target");
} else if (typeof Float16Array === "undefined") {
	throw new Error("Chromium Canvas HDR requires Float16Array");
} else {
	const canvas = document.createElement("canvas");
	canvas.width = 1;
	canvas.height = 1;
	const context = canvas.getContext("2d", {
		colorSpace: "display-p3",
		colorType: "float16",
		willReadFrequently: true,
	});
	if (!context) throw new Error("Display-P3 Float16 Canvas 2D context unavailable");
	const attributes = context.getContextAttributes();
	if (
		attributes.colorSpace !== "display-p3" ||
		attributes.colorType !== "float16"
	) {
		throw new Error("Chromium ignored the Display-P3 Float16 context request");
	}
	const source = new ImageData(
		new Float16Array([4, 0.5, 0.25, 1]),
		1,
		1,
		{ colorSpace: "display-p3", pixelFormat: "rgba-float16" },
	);
	context.putImageData(source, 0, 0);
	const readback = context.getImageData(0, 0, 1, 1, {
		colorSpace: "display-p3",
		pixelFormat: "rgba-float16",
	});
	if (!(readback.data instanceof Float16Array) || readback.data[0] <= 1) {
		throw new Error("Chromium Float16 ImageData did not preserve HDR radiance");
	}
	console.log("Software Canvas HDR Chromium probe passed");
}
