/**
 * @internal Compatibility facade. Software shading should import
 * `src/lights/runtime/lightEvaluator` directly.
 */
export {
	createLightContribution,
	evaluateLightContribution,
	type LightContribution,
	type MutableLightContribution,
	type SurfacePoint,
} from "../../lights/runtime/lightEvaluator";
