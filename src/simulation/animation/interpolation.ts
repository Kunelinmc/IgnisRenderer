import { Quaternion } from '../../maths/Quaternion'
import type { AnimationInterpolation } from '../../animation/types'
import { KeyframeTrack } from '../../animation/KeyframeTrack'

export interface SampleTrackOptions {
	isQuaternion?: boolean
}

export function sampleTrack(
	track: KeyframeTrack,
	time: number,
	options: SampleTrackOptions = {}
): number[] {
	const frameCount = track.frameCount
	if (frameCount === 0) return []
	if (frameCount === 1) return getFrameValue(track, 0)

	const times = track.times
	const clampedTime = Math.max(times[0], Math.min(time, times[frameCount - 1]))
	let right = 1
	while (right < frameCount && times[right] < clampedTime) {
		right++
	}
	if (right >= frameCount) return getFrameValue(track, frameCount - 1)
	const left = Math.max(0, right - 1)
	const t0 = times[left]
	const t1 = times[right]
	const duration = Math.max(1e-6, t1 - t0)
	const t = Math.max(0, Math.min(1, (clampedTime - t0) / duration))

	switch (track.interpolation) {
		case 'step':
			return getFrameValue(track, left)
		case 'cubic':
			return sampleCubic(track, left, right, t, duration, options.isQuaternion)
		case 'linear':
		default:
			return sampleLinear(track, left, right, t, options.isQuaternion)
	}
}

function getFrameValue(track: KeyframeTrack, frameIndex: number): number[] {
	const stride = track.valueSize
	if (track.interpolation === 'cubic') {
		const offset = frameIndex * stride * 3 + stride
		return Array.from(track.values.subarray(offset, offset + stride))
	}
	const offset = frameIndex * stride
	return Array.from(track.values.subarray(offset, offset + stride))
}

function sampleLinear(
	track: KeyframeTrack,
	left: number,
	right: number,
	t: number,
	isQuaternion?: boolean
): number[] {
	const leftValue = getFrameValue(track, left)
	const rightValue = getFrameValue(track, right)
	if (isQuaternion) {
		const q0 = new Quaternion(
			leftValue[0],
			leftValue[1],
			leftValue[2],
			leftValue[3]
		).normalize()
		const q1 = new Quaternion(
			rightValue[0],
			rightValue[1],
			rightValue[2],
			rightValue[3]
		).normalize()
		const q = Quaternion.slerp(q0, q1, t).normalize()
		return [q.x, q.y, q.z, q.w]
	}
	const result = new Array(track.valueSize)
	for (let i = 0; i < track.valueSize; i++) {
		result[i] = leftValue[i] + (rightValue[i] - leftValue[i]) * t
	}
	return result
}

function sampleCubic(
	track: KeyframeTrack,
	left: number,
	right: number,
	t: number,
	deltaTime: number,
	isQuaternion?: boolean
): number[] {
	const stride = track.valueSize
	const leftBase = left * stride * 3
	const rightBase = right * stride * 3
	const result = new Array(stride)

	const t2 = t * t
	const t3 = t2 * t
	const h00 = 2 * t3 - 3 * t2 + 1
	const h10 = t3 - 2 * t2 + t
	const h01 = -2 * t3 + 3 * t2
	const h11 = t3 - t2

	for (let i = 0; i < stride; i++) {
		const p0 = track.values[leftBase + stride + i]
		const m0 = track.values[leftBase + stride * 2 + i] * deltaTime
		const p1 = track.values[rightBase + stride + i]
		const m1 = track.values[rightBase + i] * deltaTime
		result[i] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
	}

	if (isQuaternion) {
		const q = new Quaternion(result[0], result[1], result[2], result[3]).normalize()
		return [q.x, q.y, q.z, q.w]
	}

	return result
}
