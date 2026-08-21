#version 300 es
precision highp float;
__WEBGL_SHADOW_TRANSMITTANCE_DEFINE__
#import <ignis/webgl/constants>
#import <ignis/color/srgb>
#import <ignis/postprocess/fog>

const int MAX_DIRECTIONAL_LIGHTS = __WEBGL_MAX_DIRECTIONAL_LIGHTS__;
const int MAX_POINT_LIGHTS = __WEBGL_MAX_POINT_LIGHTS__;
const int MAX_SPOT_LIGHTS = __WEBGL_MAX_SPOT_LIGHTS__;
const int MAX_LOCAL_LIGHT_PROBES = __WEBGL_MAX_LOCAL_LIGHT_PROBES__;
const int MAX_REFLECTION_PROBES = __WEBGL_MAX_REFLECTION_PROBES__;
const int MAX_CLUSTER_LIGHTS_PER_FRAGMENT =
	__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__;
const int SH_COEFFICIENT_COUNT = 16;

const float PBR_MIN_NDOTV = 0.001;
const float PBR_SPEC_FALLBACK = 0.02;
const float TRANSMISSION_ALPHA_FLOOR = 0.12;
