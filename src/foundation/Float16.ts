const FLOAT32_TO_FLOAT16_FLOAT = new Float32Array(1);
const FLOAT32_TO_FLOAT16_INT = new Int32Array(FLOAT32_TO_FLOAT16_FLOAT.buffer);

/**
 * Converts a finite 32-bit float value to IEEE 754 binary16 bits.
 *
 * @param value - The numeric value to encode.
 * @returns The unsigned 16-bit binary16 bit pattern. Non-finite values encode
 * as `0`, and overflow clamps to the largest finite binary16 value.
 */
export function float32ToFloat16Bits(value: number): number {
	if (!Number.isFinite(value)) {
		return 0;
	}
	FLOAT32_TO_FLOAT16_FLOAT[0] = value;
	const bits = FLOAT32_TO_FLOAT16_INT[0];
	const sign = (bits >> 16) & 0x8000;
	let exponent = ((bits >> 23) & 0xff) - 127 + 15;
	let mantissa = bits & 0x7fffff;

	if (exponent <= 0) {
		if (exponent < -10) {
			return sign;
		}
		mantissa = (mantissa | 0x800000) >> (1 - exponent);
		return sign | ((mantissa + 0x1000) >> 13);
	}

	if (exponent >= 31) {
		return sign | 0x7bff;
	}

	let halfMantissa = (mantissa + 0x1000) >> 13;
	if (halfMantissa === 0x400) {
		halfMantissa = 0;
		exponent++;
		if (exponent >= 31) {
			return sign | 0x7bff;
		}
	}

	return sign | (exponent << 10) | halfMantissa;
}

/**
 * Converts IEEE 754 binary16 bits to a JavaScript number.
 *
 * @param bits - The unsigned 16-bit binary16 bit pattern.
 * @returns The decoded 32-bit float value.
 */
export function float16BitsToFloat32(bits: number): number {
	const value = bits & 0xffff;
	const sign = (value & 0x8000) !== 0 ? -1 : 1;
	const exponent = (value >> 10) & 0x1f;
	const mantissa = value & 0x03ff;

	if (exponent === 0) {
		if (mantissa === 0) {
			return sign * 0;
		}
		return sign * Math.pow(2, -14) * (mantissa / 1024);
	}
	if (exponent === 0x1f) {
		return mantissa === 0 ? sign * Infinity : Number.NaN;
	}
	return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}
