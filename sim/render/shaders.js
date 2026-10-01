// GLSL for the v2 engine. Every material is a ShaderMaterial authored directly in
// display (sRGB) values with one shared lighting + haze model and no tone mapping or
// colour-space conversion. So the screen and capture() render targets produce the
// same pixels, which the fidelity metrics depend on.

const COMMON = /* glsl */ `
uniform vec3 uSunDir;
uniform vec3 uSunCol;
uniform vec3 uAmbSky;
uniform vec3 uAmbGround;
uniform vec3 uCamPos;
uniform vec3 uFogCol;
uniform vec3 uFogSun;
uniform float uFogDen;
uniform float uTime;

vec3 shade(vec3 albedo, vec3 n, float wrap) {
  float d = clamp((dot(n, uSunDir) + wrap) / (1.0 + wrap), 0.0, 1.0);
  vec3 amb = mix(uAmbGround, uAmbSky, n.y * 0.5 + 0.5);
  return albedo * (amb + uSunCol * d);
}
vec3 hazeColor(vec3 dir) {
  float s = pow(max(dot(dir, uSunDir), 0.0), 6.0);
  return mix(uFogCol, uFogSun, s);
}
vec3 applyFog(vec3 col, vec3 wpos) {
  vec3 v = wpos - uCamPos;
  float dist = length(v);
  float f = 1.0 - exp(-pow(dist * uFogDen, 1.3));
  return mix(col, hazeColor(v / max(dist, 1e-3)), f);
}
`;

// Shared varyings/attributes for the ground (near strip and far grid).
export const terrainVert = /* glsl */ `
attribute float aLat;
attribute float aD;
attribute float aForest;
varying vec3 vW; varying vec3 vN; varying float vLat; varying float vD; varying float vForest;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz; vN = normal; vLat = aLat; vD = aD; vForest = aForest;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

export const terrainFrag = COMMON + /* glsl */ `
uniform sampler2D uNoise;
uniform sampler2D uGrassTex;
uniform vec3 uGrassA; uniform vec3 uGrassB; uniform vec3 uGrassDry;
uniform vec3 uRockCol; uniform vec3 uGravelCol; uniform vec3 uAsphaltCol; uniform vec3 uLineCol;
uniform vec3 uForestCol; uniform vec3 uSandCol; uniform vec3 uSnowCol;
uniform float uSnow; uniform float uWet; uniform float uSeaLevel;
uniform float uHalfRoad; uniform float uEdgeIn; uniform float uEdgeOut; uniform float uDashHalf;
uniform float uDashPeriod; uniform float uDashOn;
varying vec3 vW; varying vec3 vN; varying float vLat; varying float vD; varying float vForest;

float band(float x, float lo, float hi, float aa) {
  return smoothstep(lo - aa, lo + aa, x) * (1.0 - smoothstep(hi - aa, hi + aa, x));
}
void main() {
  vec3 n = normalize(vN);
  float dist = length(vW - uCamPos);
  vec2 p = vW.xz;
  vec4 N1 = texture2D(uNoise, p / 520.0);
  vec4 N2 = texture2D(uNoise, p / 67.0);
  vec4 N3 = texture2D(uNoise, p / 6.1);

  // Grass: broad green/yellow patches, dry streaks, fine blade detail up close.
  vec3 grass = mix(uGrassA, uGrassB, smoothstep(0.3, 0.75, N1.r));
  grass = mix(grass, uGrassDry, smoothstep(0.55, 0.85, N2.g) * 0.55);
  float detail = texture2D(uGrassTex, p / 1.7).r;
  float near = 1.0 - smoothstep(15.0, 110.0, dist);
  grass *= mix(1.0, 0.7 + 0.6 * detail, near) * (0.92 + 0.16 * N3.b);

  // Distant forest cover, where the scattered trees have faded out.
  float forest = smoothstep(0.5, 0.72, vForest + (N2.b - 0.5) * 0.35) * smoothstep(260.0, 420.0, dist);
  grass = mix(grass, uForestCol * (0.75 + 0.5 * N3.r), forest);

  // Rock / earth on steep cuts, with horizontal strata.
  float steep = 1.0 - smoothstep(0.6, 0.8, n.y + (N2.r - 0.5) * 0.18);
  float strata = texture2D(uNoise, vec2(p.x * 0.013 + p.y * 0.013, vW.y * 0.09)).g;
  vec3 rock = uRockCol * (0.72 + 0.5 * strata) * (0.9 + 0.2 * N3.g);
  vec3 col = mix(grass, rock, steep);

  // Gravel shoulder, then asphalt with edge lines and a dashed centre line.
  float au = abs(vLat);
  float aa = max(fwidth(vLat), 1e-4) * 0.75;
  float gravel = 1.0 - smoothstep(uHalfRoad + 0.35, uHalfRoad + 0.95, au + (N3.g - 0.5) * 0.5);
  col = mix(col, uGravelCol * (0.82 + 0.35 * N3.a), gravel * (1.0 - steep * 0.6));
  float road = 1.0 - smoothstep(uHalfRoad - aa, uHalfRoad + aa, au);
  vec3 asphalt = uAsphaltCol * (0.94 + 0.1 * N3.r + 0.05 * N2.a) * (1.0 - uWet * 0.35);
  float edge = band(au, uEdgeIn, uEdgeOut, aa);
  float dd = max(fwidth(vD), 1e-4) / uDashPeriod;
  float t = fract(vD / uDashPeriod);
  float dash = smoothstep(0.0, dd, t) * (1.0 - smoothstep(uDashOn - dd, uDashOn + dd, t));
  float centre = (1.0 - smoothstep(uDashHalf - aa, uDashHalf + aa, au)) * dash;
  asphalt = mix(asphalt, uLineCol * (0.96 + 0.04 * N3.r), max(edge, centre));
  col = mix(col, asphalt, road);

  // Wet sand at the waterline, snow cover on flat ground.
  col = mix(col, uSandCol, (1.0 - smoothstep(uSeaLevel + 0.2, uSeaLevel + 1.4, vW.y)) * (1.0 - road));
  col = mix(col, uSnowCol, uSnow * smoothstep(0.55, 0.85, n.y) * (1.0 - road * 0.75));

  gl_FragColor = vec4(applyFog(shade(col, n, 0.25), vW), 1.0);
}`;

// Instanced alpha cards (canopies, conifers, bushes, grass tufts).
export const cardVert = /* glsl */ `
uniform float uTime;
uniform float uSway;
varying vec2 vUv; varying vec3 vW; varying vec3 vN; varying vec3 vTint;
void main() {
  vec4 w = modelMatrix * instanceMatrix * vec4(position, 1.0);
  // Gentle wind: upper vertices sway, phase varies over the world.
  float h = uv.y;
  w.x += sin(uTime * 1.7 + w.z * 0.21 + w.x * 0.13) * uSway * h * h;
  w.z += cos(uTime * 1.3 + w.x * 0.17) * uSway * 0.6 * h * h;
  vW = w.xyz;
  vN = normalize(mat3(modelMatrix * instanceMatrix) * normal);
  vUv = uv;
#ifdef USE_INSTANCING_COLOR
  vTint = instanceColor;
#else
  vTint = vec3(1.0);
#endif
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

export const cardFrag = COMMON + /* glsl */ `
uniform sampler2D uMap;
uniform float uWrap;
uniform float uEdge;      // alpha-edge width in pixels: wider = softer, less shimmer on thin blades
uniform float uEdgeGrow;  // extra edge width per metre of distance (thin blades blend, not pop)
uniform vec3 uSnowCol; uniform float uSnow;
varying vec2 vUv; varying vec3 vW; varying vec3 vN; varying vec3 vTint;
void main() {
  vec4 t = texture2D(uMap, vUv);
  // Sharpened alpha: crisp cut-outs that still anti-alias with alpha-to-coverage.
  float edge = uEdge * (1.0 + length(vW - uCamPos) * uEdgeGrow);
  float a = (t.a - 0.5) / max(fwidth(t.a) * edge, 1e-3) + 0.5;
  if (a <= 0.0) discard;
  vec3 n = normalize(vN);
  vec3 alb = t.rgb * vTint;
  alb = mix(alb, uSnowCol * t.r, uSnow * 0.55 * smoothstep(0.0, 0.6, n.y));
  gl_FragColor = vec4(applyFog(shade(alb, n, uWrap), vW), clamp(a, 0.0, 1.0));
}`;

// Opaque vertex-coloured meshes (car, guardrails, posts, rocks, trunks).
export const solidVert = /* glsl */ `
varying vec3 vW; varying vec3 vN; varying vec3 vCol;
void main() {
#ifdef USE_INSTANCING
  mat4 m = modelMatrix * instanceMatrix;
#else
  mat4 m = modelMatrix;
#endif
  vec4 w = m * vec4(position, 1.0);
  vW = w.xyz;
  vN = normalize(mat3(m) * normal);
#ifdef USE_COLOR
  vCol = color;
#else
  vCol = vec3(1.0);
#endif
#ifdef USE_INSTANCING_COLOR
  vCol *= instanceColor;
#endif
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

export const solidFrag = COMMON + /* glsl */ `
uniform vec3 uColor;
uniform float uSpec;
uniform float uGloss;
uniform float uWrap;
uniform float uAmbScale;
uniform float uSunScale;
varying vec3 vW; varying vec3 vN; varying vec3 vCol;
void main() {
  vec3 n = normalize(vN);
  vec3 v = normalize(uCamPos - vW);
  if (dot(n, v) < 0.0) n = -n;   // double-sided
  float d = clamp((dot(n, uSunDir) + uWrap) / (1.0 + uWrap), 0.0, 1.0);
  vec3 amb = mix(uAmbGround, uAmbSky, n.y * 0.5 + 0.5);
  vec3 c = uColor * vCol * (amb * uAmbScale + uSunCol * d * uSunScale);
  vec3 h = normalize(uSunDir + v);
  c += uSunCol * uSpec * pow(max(dot(n, h), 0.0), uGloss);
  // Sky reflection on glossy surfaces (car paint / glass): brighter at grazing angles.
  float fr = pow(1.0 - max(dot(n, v), 0.0), 4.0);
  c = mix(c, uFogCol, fr * uSpec * 0.6);
  gl_FragColor = vec4(applyFog(c, vW), 1.0);
}`;

// Sky dome: gradient + procedural clouds, horizon matched to the haze colour so the
// terrain fades into it seamlessly.
export const skyVert = /* glsl */ `
varying vec3 vW;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vW = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
  gl_Position.z = gl_Position.w;   // always at the far plane
}`;

export const skyFrag = COMMON + /* glsl */ `
uniform sampler2D uNoise;
uniform vec3 uZenith; uniform vec3 uHorizon; uniform vec3 uCloudCol; uniform vec3 uCloudShade;
uniform float uCloudCover; uniform vec2 uCloudOff; uniform vec3 uStars; uniform float uNight;
varying vec3 vW;
void main() {
  vec3 dir = normalize(vW - uCamPos);
  float h = dir.y;
  vec3 sky = mix(uHorizon, uZenith, pow(clamp(h, 0.0, 1.0), 0.55));
  // Clouds on a virtual plane: two noise octaves, soft coverage, shaded undersides.
  vec2 cp = dir.xz / max(h + 0.06, 0.04) * 0.23 + uCloudOff;
  float c = texture2D(uNoise, cp * 0.35).r * 0.62 + texture2D(uNoise, cp * 1.3).g * 0.28 + texture2D(uNoise, cp * 4.1).b * 0.1;
  float cover = smoothstep(1.0 - uCloudCover, 1.2 - uCloudCover * 0.6, c) * smoothstep(0.0, 0.16, h);
  vec3 cloud = mix(uCloudShade, uCloudCol, smoothstep(0.45, 0.85, c));
  sky = mix(sky, cloud, cover * 0.9);
  sky += uSunCol * 0.35 * pow(max(dot(dir, uSunDir), 0.0), 350.0);
  // Stars at night.
  float st = step(0.9965, texture2D(uNoise, dir.xz / max(h, 0.05) * 9.0).a) * uNight * smoothstep(0.05, 0.3, h);
  sky += uStars * st;
  // Blend to the haze colour at and below the horizon.
  sky = mix(hazeColor(dir), sky, smoothstep(-0.01, 0.12, h));
  gl_FragColor = vec4(sky, 1.0);
}`;

export const waterFrag = COMMON + /* glsl */ `
uniform sampler2D uNoise;
uniform vec3 uWater; uniform vec3 uWaterDeep;
varying vec3 vW; varying vec3 vN; varying float vLat; varying float vD; varying float vForest;
void main() {
  vec2 p = vW.xz;
  float w1 = texture2D(uNoise, p / 11.0 + vec2(uTime * 0.012, uTime * 0.007)).b;
  float w2 = texture2D(uNoise, p / 4.3 - vec2(uTime * 0.02, -uTime * 0.013)).g;
  float w = w1 * 0.6 + w2 * 0.4;
  vec3 v = normalize(uCamPos - vW);
  vec3 col = mix(uWaterDeep, uWater, 0.55 + 0.45 * w);
  col = mix(col, vec3(0.93, 0.97, 1.0), smoothstep(0.66, 0.8, w) * 0.65);   // wave crests
  float fr = pow(1.0 - max(v.y, 0.0), 5.0);
  col = mix(col, uFogCol, fr * 0.55);
  gl_FragColor = vec4(applyFog(col, vW), 1.0);
}`;

// Soft contact shadow under the car.
export const shadowVert = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

export const shadowFrag = /* glsl */ `
uniform float uStrength;
varying vec2 vUv;
void main() {
  vec2 q = abs(vUv - 0.5) * 2.0;
  float d = length(max(q - vec2(0.55, 0.7), 0.0)) / 0.45;
  float a = uStrength * (1.0 - smoothstep(0.0, 1.0, d)) * (1.0 - 0.25 * smoothstep(0.3, 1.0, max(q.x, q.y)));
  gl_FragColor = vec4(0.0, 0.0, 0.0, a);
}`;
