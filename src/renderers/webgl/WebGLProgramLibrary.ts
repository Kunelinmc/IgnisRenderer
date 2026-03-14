import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";

export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: {
		model: WebGLUniformLocation | null;
		viewProjection: WebGLUniformLocation | null;
		normalMatrix: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		ambientColor: WebGLUniformLocation | null;
		enableLighting: WebGLUniformLocation | null;
		shadingModel: WebGLUniformLocation | null;
		baseColor: WebGLUniformLocation | null;
		emissive: WebGLUniformLocation | null;
		pbr: WebGLUniformLocation | null;
		phong: WebGLUniformLocation | null;
		alpha: WebGLUniformLocation | null;
		baseMap: WebGLUniformLocation | null;
		hasBaseMap: WebGLUniformLocation | null;
		baseMapIsLinear: WebGLUniformLocation | null;
		dirLightCount: WebGLUniformLocation | null;
		dirLightDirection: WebGLUniformLocation | null;
		dirLightColor: WebGLUniformLocation | null;
		pointLightCount: WebGLUniformLocation | null;
		pointLightPositionRange: WebGLUniformLocation | null;
		pointLightColor: WebGLUniformLocation | null;
		spotLightCount: WebGLUniformLocation | null;
		spotLightPositionRange: WebGLUniformLocation | null;
		spotLightDirectionOuter: WebGLUniformLocation | null;
		spotLightColorInner: WebGLUniformLocation | null;
	};
}

export interface WebGLSkyboxProgram {
	program: WebGLProgram;
	uniforms: {
		skyboxMap: WebGLUniformLocation | null;
		skyboxBasisRight: WebGLUniformLocation | null;
		skyboxBasisUp: WebGLUniformLocation | null;
		skyboxBasisBackward: WebGLUniformLocation | null;
		skyboxIsOrthographic: WebGLUniformLocation | null;
		skyboxMapIsLinear: WebGLUniformLocation | null;
	};
}

export interface WebGLPresentProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		applyGamma: WebGLUniformLocation | null;
	};
}

export interface WebGLParticleProgram {
	program: WebGLProgram;
	uniforms: {
		viewProjection: WebGLUniformLocation | null;
		basisRight: WebGLUniformLocation | null;
		basisUp: WebGLUniformLocation | null;
		particleMap: WebGLUniformLocation | null;
		uvTransformA: WebGLUniformLocation | null;
		uvTransformB: WebGLUniformLocation | null;
		mapIsLinear: WebGLUniformLocation | null;
	};
}

export interface WebGLFXAAProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

type WarnFn = (key: string, message: string) => void;

const SCENE_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;

void main() {
	vec4 worldPos = uModel * vec4(aPosition, 1.0);
	vWorldPos = worldPos.xyz;
	vNormal = normalize(uNormalMatrix * aNormal);
	vUv = aUv;
	gl_Position = uViewProjection * worldPos;
}
`;

const SCENE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

const int MAX_DIRECTIONAL_LIGHTS = ${WEBGL_MAX_DIRECTIONAL_LIGHTS};
const int MAX_POINT_LIGHTS = ${WEBGL_MAX_POINT_LIGHTS};
const int MAX_SPOT_LIGHTS = ${WEBGL_MAX_SPOT_LIGHTS};

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUv;

uniform vec3 uCameraPosition;
uniform vec3 uAmbientColor;
uniform int uEnableLighting;
uniform int uShadingModel;
uniform vec4 uBaseColor;
uniform vec4 uEmissive;
uniform vec4 uPBR;
uniform vec4 uPhong;
uniform vec4 uAlpha;
uniform sampler2D uBaseMap;
uniform int uHasBaseMap;
uniform int uBaseMapIsLinear;

uniform int uDirLightCount;
uniform vec4 uDirLightDirection[MAX_DIRECTIONAL_LIGHTS];
uniform vec4 uDirLightColor[MAX_DIRECTIONAL_LIGHTS];

uniform int uPointLightCount;
uniform vec4 uPointLightPositionRange[MAX_POINT_LIGHTS];
uniform vec4 uPointLightColor[MAX_POINT_LIGHTS];

uniform int uSpotLightCount;
uniform vec4 uSpotLightPositionRange[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightDirectionOuter[MAX_SPOT_LIGHTS];
uniform vec4 uSpotLightColorInner[MAX_SPOT_LIGHTS];

out vec4 fragColor;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

vec3 shadePhong(vec3 albedo, vec3 n, vec3 v) {
	vec3 lit = uAmbientColor * albedo;
	vec3 specular = vec3(0.0);
	float shininess = max(1.0, uPhong.x);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = normalize(uDirLightDirection[i].xyz);
		float nDotL = max(dot(n, l), 0.0);
		lit += albedo * uDirLightColor[i].xyz * nDotL;
		if (nDotL > 0.0) {
			vec3 h = normalize(l + v);
			specular += uDirLightColor[i].xyz * pow(max(dot(n, h), 0.0), shininess);
		}
	}

	for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
		if (i >= uPointLightCount) break;
		vec3 delta = uPointLightPositionRange[i].xyz - vWorldPos;
		float distance = max(length(delta), 0.0001);
		vec3 l = delta / distance;
		float attenuation = 1.0 / (1.0 + distance * distance / max(uPointLightPositionRange[i].w, 0.001));
		float nDotL = max(dot(n, l), 0.0);
		lit += albedo * uPointLightColor[i].xyz * nDotL * attenuation;
		if (nDotL > 0.0) {
			vec3 h = normalize(l + v);
			specular += uPointLightColor[i].xyz * pow(max(dot(n, h), 0.0), shininess) * attenuation;
		}
	}

	for (int i = 0; i < MAX_SPOT_LIGHTS; i++) {
		if (i >= uSpotLightCount) break;
		vec3 delta = uSpotLightPositionRange[i].xyz - vWorldPos;
		float distance = max(length(delta), 0.0001);
		vec3 l = delta / distance;
		float attenuation = 1.0 / (1.0 + distance * distance / max(uSpotLightPositionRange[i].w, 0.001));
		vec3 coneDir = normalize(-uSpotLightDirectionOuter[i].xyz);
		float coneCos = dot(l, coneDir);
		float coneFactor = smoothstep(uSpotLightDirectionOuter[i].w, uSpotLightColorInner[i].w, coneCos);
		float nDotL = max(dot(n, l), 0.0);
		lit += albedo * uSpotLightColorInner[i].xyz * nDotL * attenuation * coneFactor;
		if (nDotL > 0.0) {
			vec3 h = normalize(l + v);
			specular += uSpotLightColorInner[i].xyz * pow(max(dot(n, h), 0.0), shininess) * attenuation * coneFactor;
		}
	}

	return lit + specular * 0.25;
}

vec3 shadePBR(vec3 albedo, vec3 n, vec3 v) {
	vec3 lit = uAmbientColor * albedo;
	float roughness = clamp(uPBR.x, 0.04, 1.0);
	float metalness = clamp(uPBR.y, 0.0, 1.0);
	float reflectance = clamp(uPBR.z, 0.0, 1.0);
	float dielectricF0 = 0.16 * reflectance * reflectance;
	vec3 f0 = mix(vec3(dielectricF0), albedo, metalness);
	float specPower = mix(128.0, 8.0, roughness);

	for (int i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
		if (i >= uDirLightCount) break;
		vec3 l = normalize(uDirLightDirection[i].xyz);
		float nDotL = max(dot(n, l), 0.0);
		if (nDotL <= 0.0) continue;
		vec3 h = normalize(l + v);
		float nDotV = max(dot(n, v), 0.001);
		float hDotV = max(dot(h, v), 0.0);
		vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - hDotV, 5.0);
		vec3 diffuse = (1.0 - metalness) * albedo / 3.14159265;
		vec3 specular = fresnel * pow(max(dot(n, h), 0.0), specPower);
		lit += (diffuse + specular) * uDirLightColor[i].xyz * nDotL * nDotV;
	}

	for (int i = 0; i < MAX_POINT_LIGHTS; i++) {
		if (i >= uPointLightCount) break;
		vec3 delta = uPointLightPositionRange[i].xyz - vWorldPos;
		float distance = max(length(delta), 0.0001);
		vec3 l = delta / distance;
		float attenuation = 1.0 / (1.0 + distance * distance / max(uPointLightPositionRange[i].w, 0.001));
		float nDotL = max(dot(n, l), 0.0);
		if (nDotL <= 0.0) continue;
		vec3 h = normalize(l + v);
		float hDotV = max(dot(h, v), 0.0);
		vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - hDotV, 5.0);
		vec3 diffuse = (1.0 - metalness) * albedo / 3.14159265;
		vec3 specular = fresnel * pow(max(dot(n, h), 0.0), specPower);
		lit += (diffuse + specular) * uPointLightColor[i].xyz * nDotL * attenuation;
	}

	for (int i = 0; i < MAX_SPOT_LIGHTS; i++) {
		if (i >= uSpotLightCount) break;
		vec3 delta = uSpotLightPositionRange[i].xyz - vWorldPos;
		float distance = max(length(delta), 0.0001);
		vec3 l = delta / distance;
		float attenuation = 1.0 / (1.0 + distance * distance / max(uSpotLightPositionRange[i].w, 0.001));
		vec3 coneDir = normalize(-uSpotLightDirectionOuter[i].xyz);
		float coneCos = dot(l, coneDir);
		float coneFactor = smoothstep(uSpotLightDirectionOuter[i].w, uSpotLightColorInner[i].w, coneCos);
		float nDotL = max(dot(n, l), 0.0);
		if (nDotL <= 0.0 || coneFactor <= 0.0) continue;
		vec3 h = normalize(l + v);
		float hDotV = max(dot(h, v), 0.0);
		vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - hDotV, 5.0);
		vec3 diffuse = (1.0 - metalness) * albedo / 3.14159265;
		vec3 specular = fresnel * pow(max(dot(n, h), 0.0), specPower);
		lit += (diffuse + specular) * uSpotLightColorInner[i].xyz * nDotL * attenuation * coneFactor;
	}

	return lit;
}

void main() {
	vec3 albedo = uBaseColor.rgb;
	float alpha = clamp(uBaseColor.a, 0.0, 1.0);
	if (uHasBaseMap == 1) {
		vec4 texel = texture(uBaseMap, vUv);
		vec3 texColor = uBaseMapIsLinear == 1 ? texel.rgb : srgbToLinear(texel.rgb);
		albedo *= texColor;
		alpha *= texel.a;
	}

	if (uAlpha.y > 0.5 && alpha < uAlpha.x) {
		discard;
	}

	vec3 normal = normalize(vNormal);
	vec3 viewDir = normalize(uCameraPosition - vWorldPos);
	vec3 color;
	if (uEnableLighting == 0 || uShadingModel == 2) {
		color = albedo;
	} else if (uShadingModel == 1) {
		color = shadePBR(albedo, normal, viewDir);
	} else {
		color = shadePhong(albedo, normal, viewDir);
	}

	color += uEmissive.rgb;
	fragColor = vec4(max(color, vec3(0.0)), alpha);
}
`;

const SKYBOX_VERTEX_SHADER = `#version 300 es
precision highp float;

out vec2 vNdc;

void main() {
	vec2 pos;
	if (gl_VertexID == 0) {
		pos = vec2(-1.0, -1.0);
	} else if (gl_VertexID == 1) {
		pos = vec2(3.0, -1.0);
	} else {
		pos = vec2(-1.0, 3.0);
	}
	vNdc = pos;
	gl_Position = vec4(pos, 0.0, 1.0);
}
`;

const SKYBOX_FRAGMENT_SHADER = `#version 300 es
precision highp float;

const float PI = 3.14159265359;

in vec2 vNdc;

uniform sampler2D uSkyboxMap;
uniform vec4 uSkyboxBasisRight;
uniform vec4 uSkyboxBasisUp;
uniform vec3 uSkyboxBasisBackward;
uniform float uSkyboxIsOrthographic;
uniform int uSkyboxMapIsLinear;

out vec4 fragColor;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

void main() {
	vec3 right = uSkyboxBasisRight.xyz;
	vec3 up = uSkyboxBasisUp.xyz;
	vec3 backward = uSkyboxBasisBackward.xyz;
	float tanHalfFov = uSkyboxBasisRight.w;
	float aspect = uSkyboxBasisUp.w;

	vec3 dir;
	if (uSkyboxIsOrthographic > 0.5) {
		dir = normalize(-backward);
	} else {
		float cx = vNdc.x * aspect * tanHalfFov;
		float cy = vNdc.y * tanHalfFov;
		dir = normalize(right * cx + up * cy - backward);
	}

	float phi = atan(dir.x, dir.z);
	float theta = acos(clamp(dir.y, -1.0, 1.0));
	vec2 uv = vec2((phi + PI) / (2.0 * PI), theta / PI);
	vec4 sampled = texture(uSkyboxMap, uv);
	vec3 sky = uSkyboxMapIsLinear == 1 ? sampled.rgb : srgbToLinear(sampled.rgb);
	fragColor = vec4(max(sky, vec3(0.0)), 1.0);
}
`;

const PRESENT_VERTEX_SHADER = `#version 300 es
precision highp float;

out vec2 vUv;

void main() {
	vec2 pos;
	if (gl_VertexID == 0) {
		pos = vec2(-1.0, -1.0);
	} else if (gl_VertexID == 1) {
		pos = vec2(3.0, -1.0);
	} else {
		pos = vec2(-1.0, 3.0);
	}
	gl_Position = vec4(pos, 0.0, 1.0);
	vUv = vec2(pos.x * 0.5 + 0.5, pos.y * 0.5 + 0.5);
}
`;

const PRESENT_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform int uApplyGamma;

out vec4 fragColor;

vec3 linearToSrgb(vec3 c) {
	vec3 a = c * 12.92;
	vec3 b = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
	return mix(b, a, lessThanEqual(c, vec3(0.0031308)));
}

void main() {
	vec4 sampled = texture(uSourceMap, vUv);
	vec3 color = sampled.rgb;
	if (uApplyGamma == 1) {
		color = linearToSrgb(max(color, vec3(0.0)));
	}
	fragColor = vec4(clamp(color, 0.0, 1.0), sampled.a);
}
`;

const PARTICLE_VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aQuadPosition;
layout(location = 1) in vec2 aQuadUv;
layout(location = 2) in vec4 aInstancePositionSize;
layout(location = 3) in vec4 aInstanceColor;
layout(location = 4) in vec4 aInstanceUvRect;
layout(location = 5) in float aInstanceRotation;

uniform mat4 uViewProjection;
uniform vec3 uBasisRight;
uniform vec3 uBasisUp;

out vec2 vUv;
out vec4 vColor;
out vec2 vLocalUv;

void main() {
	float c = cos(aInstanceRotation);
	float s = sin(aInstanceRotation);
	vec2 rotated = vec2(
		aQuadPosition.x * c - aQuadPosition.y * s,
		aQuadPosition.x * s + aQuadPosition.y * c
	);
	vec3 worldPosition =
		aInstancePositionSize.xyz +
		(uBasisRight * rotated.x + uBasisUp * rotated.y) *
			aInstancePositionSize.w;

	gl_Position = uViewProjection * vec4(worldPosition, 1.0);
	vUv = vec2(
		mix(aInstanceUvRect.x, aInstanceUvRect.z, aQuadUv.x),
		mix(aInstanceUvRect.y, aInstanceUvRect.w, aQuadUv.y)
	);
	vColor = aInstanceColor;
	vLocalUv = aQuadUv;
}
`;

const PARTICLE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;
in vec4 vColor;
in vec2 vLocalUv;

uniform sampler2D uParticleMap;
uniform vec4 uUvTransformA;
uniform vec2 uUvTransformB;
uniform int uMapIsLinear;

out vec4 fragColor;

vec3 srgbToLinear(vec3 c) {
	vec3 a = c / 12.92;
	vec3 b = pow((c + 0.055) / 1.055, vec3(2.4));
	return mix(b, a, lessThanEqual(c, vec3(0.04045)));
}

void main() {
	float radialDistance = distance(vLocalUv, vec2(0.5, 0.5));
	float radialMask = 1.0 - smoothstep(0.4, 0.5, radialDistance);

	vec2 scaledUv = vUv * uUvTransformA.xy;
	vec2 rotatedUv = vec2(
		scaledUv.x * uUvTransformB.x - scaledUv.y * uUvTransformB.y,
		scaledUv.x * uUvTransformB.y + scaledUv.y * uUvTransformB.x
	);
	vec2 finalUv = rotatedUv + uUvTransformA.zw;
	vec4 sampled = texture(uParticleMap, finalUv);
	if (uMapIsLinear == 0) {
		sampled.rgb = srgbToLinear(sampled.rgb);
	}

	vec4 color = sampled * vColor;
	color.a *= radialMask;
	if (color.a <= 0.001) {
		discard;
	}

	fragColor = vec4(max(color.rgb, vec3(0.0)), clamp(color.a, 0.0, 1.0));
}
`;

const FXAA_VERTEX_SHADER = PRESENT_VERTEX_SHADER;

const FXAA_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;

uniform sampler2D uSourceMap;
uniform vec2 uTexelSize;

out vec4 fragColor;

float luma(vec3 color) {
	return dot(color, vec3(0.299, 0.587, 0.114));
}

void main() {
	vec3 rgbNW = texture(uSourceMap, vUv + vec2(-1.0, -1.0) * uTexelSize).rgb;
	vec3 rgbNE = texture(uSourceMap, vUv + vec2(1.0, -1.0) * uTexelSize).rgb;
	vec3 rgbSW = texture(uSourceMap, vUv + vec2(-1.0, 1.0) * uTexelSize).rgb;
	vec3 rgbSE = texture(uSourceMap, vUv + vec2(1.0, 1.0) * uTexelSize).rgb;
	vec3 rgbM = texture(uSourceMap, vUv).rgb;

	float lumaNW = luma(rgbNW);
	float lumaNE = luma(rgbNE);
	float lumaSW = luma(rgbSW);
	float lumaSE = luma(rgbSE);
	float lumaM = luma(rgbM);

	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y = (lumaNW + lumaSW) - (lumaNE + lumaSE);

	float dirReduce = max(
		(lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * 0.03125),
		0.0078125
	);
	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * uTexelSize;

	vec3 rgbA = 0.5 * (
		texture(uSourceMap, vUv + dir * (1.0 / 3.0 - 0.5)).rgb +
		texture(uSourceMap, vUv + dir * (2.0 / 3.0 - 0.5)).rgb
	);
	vec3 rgbB = rgbA * 0.5 + 0.25 * (
		texture(uSourceMap, vUv + dir * -0.5).rgb +
		texture(uSourceMap, vUv + dir * 0.5).rgb
	);

	float lumaB = luma(rgbB);
	vec3 filtered =
		(lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
	fragColor = vec4(max(filtered, vec3(0.0)), 1.0);
}
`;

export class WebGLProgramLibrary {
	private _gl: WebGL2RenderingContext;
	private _warn: WarnFn;
	private _sceneProgram: WebGLSceneProgram | null = null;
	private _skyboxProgram: WebGLSkyboxProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _fxaaProgram: WebGLFXAAProgram | null = null;

	constructor(gl: WebGL2RenderingContext, warn: WarnFn) {
		this._gl = gl;
		this._warn = warn;
	}

	public getSceneProgram(): WebGLSceneProgram {
		if (this._sceneProgram) {
			return this._sceneProgram;
		}
		const program = this._createProgram(
			SCENE_VERTEX_SHADER,
			SCENE_FRAGMENT_SHADER,
			"WebGLSceneProgram"
		);
		this._sceneProgram = {
			program,
			uniforms: {
				model: this._gl.getUniformLocation(program, "uModel"),
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				normalMatrix: this._gl.getUniformLocation(program, "uNormalMatrix"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				ambientColor: this._gl.getUniformLocation(program, "uAmbientColor"),
				enableLighting: this._gl.getUniformLocation(program, "uEnableLighting"),
				shadingModel: this._gl.getUniformLocation(program, "uShadingModel"),
				baseColor: this._gl.getUniformLocation(program, "uBaseColor"),
				emissive: this._gl.getUniformLocation(program, "uEmissive"),
				pbr: this._gl.getUniformLocation(program, "uPBR"),
				phong: this._gl.getUniformLocation(program, "uPhong"),
				alpha: this._gl.getUniformLocation(program, "uAlpha"),
				baseMap: this._gl.getUniformLocation(program, "uBaseMap"),
				hasBaseMap: this._gl.getUniformLocation(program, "uHasBaseMap"),
				baseMapIsLinear: this._gl.getUniformLocation(
					program,
					"uBaseMapIsLinear"
				),
				dirLightCount: this._gl.getUniformLocation(program, "uDirLightCount"),
				dirLightDirection: this._gl.getUniformLocation(
					program,
					"uDirLightDirection"
				),
				dirLightColor: this._gl.getUniformLocation(program, "uDirLightColor"),
				pointLightCount: this._gl.getUniformLocation(
					program,
					"uPointLightCount"
				),
				pointLightPositionRange: this._gl.getUniformLocation(
					program,
					"uPointLightPositionRange"
				),
				pointLightColor: this._gl.getUniformLocation(
					program,
					"uPointLightColor"
				),
				spotLightCount: this._gl.getUniformLocation(program, "uSpotLightCount"),
				spotLightPositionRange: this._gl.getUniformLocation(
					program,
					"uSpotLightPositionRange"
				),
				spotLightDirectionOuter: this._gl.getUniformLocation(
					program,
					"uSpotLightDirectionOuter"
				),
				spotLightColorInner: this._gl.getUniformLocation(
					program,
					"uSpotLightColorInner"
				),
			},
		};
		return this._sceneProgram;
	}

	public getSkyboxProgram(): WebGLSkyboxProgram {
		if (this._skyboxProgram) {
			return this._skyboxProgram;
		}
		const program = this._createProgram(
			SKYBOX_VERTEX_SHADER,
			SKYBOX_FRAGMENT_SHADER,
			"WebGLSkyboxProgram"
		);
		this._skyboxProgram = {
			program,
			uniforms: {
				skyboxMap: this._gl.getUniformLocation(program, "uSkyboxMap"),
				skyboxBasisRight: this._gl.getUniformLocation(
					program,
					"uSkyboxBasisRight"
				),
				skyboxBasisUp: this._gl.getUniformLocation(program, "uSkyboxBasisUp"),
				skyboxBasisBackward: this._gl.getUniformLocation(
					program,
					"uSkyboxBasisBackward"
				),
				skyboxIsOrthographic: this._gl.getUniformLocation(
					program,
					"uSkyboxIsOrthographic"
				),
				skyboxMapIsLinear: this._gl.getUniformLocation(
					program,
					"uSkyboxMapIsLinear"
				),
			},
		};
		return this._skyboxProgram;
	}

	public getPresentProgram(): WebGLPresentProgram {
		if (this._presentProgram) {
			return this._presentProgram;
		}
		const program = this._createProgram(
			PRESENT_VERTEX_SHADER,
			PRESENT_FRAGMENT_SHADER,
			"WebGLPresentProgram"
		);
		this._presentProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				applyGamma: this._gl.getUniformLocation(program, "uApplyGamma"),
			},
		};
		return this._presentProgram;
	}

	public getParticleProgram(): WebGLParticleProgram {
		if (this._particleProgram) {
			return this._particleProgram;
		}
		const program = this._createProgram(
			PARTICLE_VERTEX_SHADER,
			PARTICLE_FRAGMENT_SHADER,
			"WebGLParticleProgram"
		);
		this._particleProgram = {
			program,
			uniforms: {
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				particleMap: this._gl.getUniformLocation(program, "uParticleMap"),
				uvTransformA: this._gl.getUniformLocation(program, "uUvTransformA"),
				uvTransformB: this._gl.getUniformLocation(program, "uUvTransformB"),
				mapIsLinear: this._gl.getUniformLocation(program, "uMapIsLinear"),
			},
		};
		return this._particleProgram;
	}

	public getFXAAProgram(): WebGLFXAAProgram {
		if (this._fxaaProgram) {
			return this._fxaaProgram;
		}
		const program = this._createProgram(
			FXAA_VERTEX_SHADER,
			FXAA_FRAGMENT_SHADER,
			"WebGLFXAAProgram"
		);
		this._fxaaProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._fxaaProgram;
	}

	public destroy(): void {
		if (this._sceneProgram) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		if (this._skyboxProgram) {
			this._gl.deleteProgram(this._skyboxProgram.program);
			this._skyboxProgram = null;
		}
		if (this._presentProgram) {
			this._gl.deleteProgram(this._presentProgram.program);
			this._presentProgram = null;
		}
		if (this._particleProgram) {
			this._gl.deleteProgram(this._particleProgram.program);
			this._particleProgram = null;
		}
		if (this._fxaaProgram) {
			this._gl.deleteProgram(this._fxaaProgram.program);
			this._fxaaProgram = null;
		}
	}

	private _createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string
	): WebGLProgram {
		const gl = this._gl;
		const vertexShader = this._compileShader(
			gl.VERTEX_SHADER,
			vertexSource,
			`${label}:vertex`
		);
		const fragmentShader = this._compileShader(
			gl.FRAGMENT_SHADER,
			fragmentSource,
			`${label}:fragment`
		);
		const program = gl.createProgram();
		if (!program) {
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
			throw new Error(`Failed to create WebGL program (${label})`);
		}

		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		gl.deleteShader(vertexShader);
		gl.deleteShader(fragmentShader);

		const linked = !!gl.getProgramParameter(program, gl.LINK_STATUS);
		if (!linked) {
			const log = gl.getProgramInfoLog(program) || "No program link log";
			gl.deleteProgram(program);
			throw new Error(`WebGL program link failed (${label}): ${log}`);
		}

		const validateStatus = gl.getProgramParameter(program, gl.VALIDATE_STATUS);
		if (validateStatus === false) {
			this._warn(
				`webgl-program-validate-${label}`,
				`WebGL program validation reported issues (${label}): ${gl.getProgramInfoLog(program) || "no log"}`
			);
		}

		return program;
	}

	private _compileShader(
		type: number,
		source: string,
		label: string
	): WebGLShader {
		const gl = this._gl;
		const shader = gl.createShader(type);
		if (!shader) {
			throw new Error(`Failed to create WebGL shader (${label})`);
		}
		gl.shaderSource(shader, source);
		gl.compileShader(shader);
		const compiled = !!gl.getShaderParameter(shader, gl.COMPILE_STATUS);
		if (!compiled) {
			const log = gl.getShaderInfoLog(shader) || "No shader compile log";
			gl.deleteShader(shader);
			throw new Error(`WebGL shader compile failed (${label}): ${log}`);
		}
		return shader;
	}
}
