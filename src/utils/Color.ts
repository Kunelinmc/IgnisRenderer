/**
 * Color utility functions
 */

export interface RGB {
	r: number;
	g: number;
	b: number;
}

export interface RGBA extends RGB {
	a: number;
}

export interface HSL {
	h: number;
	s: number;
	l: number;
}

export interface HSLA extends HSL {
	a: number;
}

export type ColorInput = string | Partial<HSLA> | null | undefined;

function isHslaLike(value: ColorInput): value is Partial<HSLA> {
	return !!value && typeof value === "object" && "h" in value;
}

export function rgbToHsl(r: number, g: number, b: number): HSL {
	const max = Math.max(r, g, b),
		min = Math.min(r, g, b);
	let h: number,
		s: number,
		l = (max + min) / 2;

	if (max === min) {
		h = s = 0;
	} else {
		const d = max - min;
		s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
		switch (max) {
			case r:
				h = (g - b) / d + (g < b ? 6 : 0);
				break;
			case g:
				h = (b - r) / d + 2;
				break;
			case b:
				h = (r - g) / d + 4;
				break;
			default:
				h = 0;
		}
		h *= 60;
	}

	return { h, s: s * 100, l: l * 100 };
}

export function parseColor(color: ColorInput): HSLA {
	let h: number,
		s: number,
		l: number,
		a = 1;

	if (typeof color === "string") {
		const hslMatch = color.match(
			/hsla?\(\s*([0-9.\-]+)\s*,\s*([0-9.\-]+)%\s*,\s*([0-9.\-]+)%\s*(?:,\s*([0-9.\-]+)\s*)?\)/i
		);
		if (hslMatch) {
			h = Number(hslMatch[1]);
			s = Number(hslMatch[2]);
			l = Number(hslMatch[3]);
			a = hslMatch[4] !== undefined ? Number(hslMatch[4]) : 1;
		} else {
			const rgbMatch = color.match(/rgba?\(\s*([^\)]+)\)/i);
			if (rgbMatch) {
				const parts = rgbMatch[1].split(",").map((p) => Number(p.trim()));
				const [r, g, b] = parts.map((val) => val / 255);
				a = parts[3] !== undefined ? parts[3] : 1;
				const hsl = rgbToHsl(r, g, b);
				h = hsl.h;
				s = hsl.s;
				l = hsl.l;
			} else {
				// Default blue
				h = 200;
				s = 40;
				l = 60;
			}
		}
	} else if (isHslaLike(color)) {
		h = color.h ?? 200;
		s = color.s ?? 40;
		l = color.l ?? 60;
		a = color.a ?? 1;
	} else {
		h = 200;
		s = 40;
		l = 60;
	}

	return { h, s, l, a };
}

export function adjustColorLightness(
	baseColor: ColorInput,
	intensity = 1,
	shadingFactor = 1
): HSLA {
	const { h, s, l, a } = parseColor(baseColor);
	const adjustedL = Math.max(0, Math.min(100, l * intensity * shadingFactor));
	return { h, s, l: adjustedL, a };
}

export function hslToRgb(h: number, s: number, l: number, a = 1): RGBA {
	h = h % 360;
	if (h < 0) h += 360;

	s = Math.max(0, Math.min(100, s)) / 100;
	l = Math.max(0, Math.min(100, l)) / 100;

	const c = (1 - Math.abs(2 * l - 1)) * s;
	const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
	const m = l - c / 2;

	let r: number, g: number, b: number;

	if (h >= 0 && h < 60) {
		r = c;
		g = x;
		b = 0;
	} else if (h >= 60 && h < 120) {
		r = x;
		g = c;
		b = 0;
	} else if (h >= 120 && h < 180) {
		r = 0;
		g = c;
		b = x;
	} else if (h >= 180 && h < 240) {
		r = 0;
		g = x;
		b = c;
	} else if (h >= 240 && h < 300) {
		r = x;
		g = 0;
		b = c;
	} else {
		r = c;
		g = 0;
		b = x;
	}

	r = Math.round((r + m) * 255);
	g = Math.round((g + m) * 255);
	b = Math.round((b + m) * 255);

	return { r, g, b, a };
}
