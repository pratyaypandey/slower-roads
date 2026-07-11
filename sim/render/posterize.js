// Temporally-stable posterize post-pass (SIM.md §2).
//
// The scene is rendered to a low-res target, then this fullscreen pass quantizes
// colour to a few bands — the "digital dream" identity, and the single biggest lever
// on frame entropy (which is what makes M1 reconstruction and M4 decode cheap).
//
// Naive per-channel flooring flickers when smooth lighting crosses a band edge (a
// pixel dithers between two bands frame to frame), injecting fake high-frequency
// motion the dynamics model would waste capacity on. We damp that with a small
// smoothstep around each band edge ("soft posterize"): values snap to bands but cross
// edges continuously, so a slowly-changing input yields a slowly-changing output.

import * as THREE from 'three';

export const POSTERIZE_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D tDiffuse;
  uniform float levels;     // colour bands per channel
  uniform float softness;   // 0 = hard bands, ~0.5 = smoothly crossed edges
  uniform float saturation; // >1 pushes the dream palette
  varying vec2 vUv;

  vec3 softPosterize(vec3 c, float n, float soft) {
    vec3 scaled = c * n;
    vec3 lower = floor(scaled);
    vec3 f = scaled - lower;                 // fractional position within a band
    // Hard step is step(0.5, f); soften the transition to kill temporal flicker.
    vec3 stepped = smoothstep(0.5 - soft, 0.5 + soft, f);
    return (lower + stepped) / n;
  }

  void main() {
    vec3 c = texture2D(tDiffuse, vUv).rgb;
    // Gentle saturation push before banding for a bolder palette.
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = clamp(mix(vec3(l), c, saturation), 0.0, 1.0);
    c = softPosterize(c, levels, softness);
    gl_FragColor = vec4(c, 1.0);
  }
`;

export const POSTERIZE_VERT = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/** Builds the fullscreen posterize quad + its own ortho scene/camera. */
export function makePosterizePass({ levels = 6, softness = 0.06, saturation = 1.15 } = {}) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      tDiffuse: { value: null },
      levels: { value: levels },
      softness: { value: softness },
      saturation: { value: saturation },
    },
    vertexShader: POSTERIZE_VERT,
    fragmentShader: POSTERIZE_FRAG,
    depthTest: false,
    depthWrite: false,
  });
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(quad);
  return { scene, camera, material };
}
