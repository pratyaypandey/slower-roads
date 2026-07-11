# Simulator Upgrade Plan

## Purpose

Upgrade `sim/` from a technically complete deterministic oracle into a compelling endless scenic drive, while preserving the properties that make the larger Slower Roads project defensible:

- deterministic state from `seed + dial schedule + action sequence`;
- cheap enough to generate training data at high throughput;
- simple and structured enough for a small world model to learn;
- explicit factors for contrastive activation steering;
- a skeleton/anchor path cheap enough to run beside on-device inference;
- low input latency and stable frame pacing.

The target is not photorealism. The target is a legible, toy-like, low-poly world with the compositional clarity and iconic silhouettes associated with games such as Polytopia, adapted to a continuous chase-camera drive. It should feel calm, surprising, and intentionally composed rather than like noise sampled forever.

The central recommendation is:

> Stop treating the world as one continuously randomized surface. Generate a deterministic sequence of authored scenic beats, render it through cached distance bands, and make the car/camera communicate the road clearly.

This document is based on `ROADMAP.md`, `plans/SIM.md`, the current implementation, and hands-on inspection of `http://localhost:8877/demo/` at desktop resolution.

---

## 1. Current-state assessment

### What is already good

The simulator has more of the right foundation than its first impression suggests:

- The deterministic core is isolated from Three.js.
- Physics uses a fixed `1/30` step.
- Road, terrain, scatter, collisions, dials, snapshots, and replay are seed-driven.
- Instancing is already used for vegetation.
- The renderer has a physical sky, fog, shadows, water, weather particles, bloom, tone mapping, and a low-poly vocabulary.
- Environment dials are continuous and smoothed.
- The renderer and future skeleton head share a useful state contract.
- The sim already sustains a high refresh rate on the development machine in the default scene. A short browser sample reported roughly 120 presented frames per second, so there is room to improve quality if frame-time variance and future model coexistence remain controlled.

These should be evolved, not replaced wholesale.

### Why it does not yet feel like a great drive

#### 1. Composition is weak

In the default view, the vehicle can dominate the lower center of the frame and obscure the most useful part of the road. The camera follows car position and heading exactly, with no suspension, lag, lateral framing, curvature look-ahead, or horizon composition. The result reads as a debug chase camera.

The distant half of the image is frequently pale sky and fog with little silhouette structure. Most detail occupies one near/mid-distance strip. The scene therefore lacks the foreground/midground/background layering that makes a vista feel deep.

#### 2. Infinite currently means uniform

Road curvature is one fBm signal; terrain is one two-scale fBm field; scatter uses one repeated road-relative lattice and a small set of primitives. Those systems vary locally, but the grammar does not change. There are no valleys, passes, coast approaches, forest clearings, plateaus, settlements, bridges, tunnels, landmark reveals, or quiet intervals as meaningful categories.

Randomness produces difference, but not memory. A strong endless game needs recurrence at the texture scale and novelty at the structural scale.

#### 3. The road is generated as geometry, not designed as a driving experience

The current road integrates noisy curvature directly and samples height directly from the terrain. It has a fixed width and no route class, shoulder variation, banking, cut/fill, bridges, guardrails, turn language, or grade constraints. Vertical changes can inherit short terrain features instead of following an engineered grade profile.

A scenic road should be the authored spine of the world. Terrain should explain and frame the route, while the route should respond plausibly to terrain.

#### 4. The driving model exposes too little state

The car has longitudinal speed, yaw, lateral slip, and a basic airborne state. This is a sound small oracle, but the player receives little feedback about load transfer, suspension, traction, surface, acceleration, or road camber. Steering is also smoothed per rendered frame in the demo, so control response changes with display refresh rate.

#### 5. Art direction is a palette blend, not yet a style grammar

The two biome values currently change colors, density, and species probabilities. They do not change landform language, road treatment, prop families, skyline shapes, lighting ratios, weather behavior, or ambient motion. Broadleaf spheres, pine cones, rocks, bushes, and grass remain recognizable programmer primitives.

The low-poly direction is right. It needs stronger shape design, controlled palettes, scale hierarchy, and biome-specific kits.

#### 6. Expensive work is repeated every frame

Current hot-path work includes:

- rebuilding every terrain vertex and color;
- multiple fBm evaluations per terrain vertex;
- recomputing terrain normals and bounds;
- regenerating all scatter objects into new arrays;
- rewriting every vegetation instance matrix and bound;
- disabling frustum culling on terrain and all vegetation groups;
- rendering a 2048-square shadow map with many instances;
- CPU-updating every precipitation particle;
- allocating temporary vectors, matrices, and colors in common update paths;
- running half-float MSAA, bloom, vignette, and tone mapping at display resolution.

The default scene can afford this today, but M4 expects the renderer/anchor to share a tight frame with neural inference. More scenery should come from reuse and scheduling, not more per-frame work.

#### 7. Some intended features are disconnected

`Road.props()` already describes monolith, arch, tree, and crystal landmark candidates, but the current renderer does not consume `sim.props()`. This is a useful seed for a landmark layer, but landmarks need spatial rules and pacing rather than simply being rendered at every candidate.

---

## 2. Experience pillars

Every proposed feature should support at least one pillar and preserve the roadmap's research requirements.

### A. The road is always readable

The player should see enough road to understand the next decision. Curves, crests, shoulders, and surface changes need advance cues. The car must not hide the route.

### B. The world has scenic rhythm

The drive alternates enclosure and openness, calm and spectacle, anticipation and reveal. A landmark is more effective after a quiet stretch than after five other landmarks.

### C. Every biome has a silhouette

Biomes must be identifiable in grayscale and at 128px, not only by color. Conifer spires, mesa shelves, coastal stacks, rounded orchard canopies, crystalline fins, and snow caps should remain legible after downsampling.

### D. Motion makes the world feel alive

Small coherent motion is higher value than more static geometry: grass lean, canopy sway, cloud parallax, rain streaks, wheel travel, body settle, dust, insects, distant birds, water shimmer, and changing wind.

### E. Surprise is structured and reproducible

All macro events are keyed by seed and road distance. Replaying the same episode must reproduce the same scenic sequence exactly.

### F. Beauty fits the model

Favor clean shapes, stable edges, large tonal regions, sparse details, and smooth temporal changes. Avoid noisy textures, aggressive temporal effects, thin subpixel geometry, and uncontrolled particle clutter.

---

## 3. Proposed world architecture

### 3.1 Separate world truth from presentation

Introduce three explicit layers:

1. **Oracle state:** car, road, terrain constraints, collisions, environment factors, and semantic object records. Fully serializable and deterministic.
2. **World plan:** deterministic, distance-addressable chunk descriptors and scenic beats. This is generated ahead and cached, but can always be regenerated from the seed.
3. **Presentation state:** camera springs, particles, audio voices, LOD fades, and quality settings. Deterministic for exported training frames where needed, but excluded from physical oracle metrics.

Do not make Three.js objects the source of truth. Renderers should materialize a world plan.

### 3.2 Add a deterministic scenic director

Generate a low-frequency sequence along road distance, for example in 250-800 m beats. Each beat has an intention:

```js
{
  id,
  d0,
  d1,
  role: 'rest' | 'tease' | 'transition' | 'reveal' | 'landmark' | 'cooldown',
  biomeWeights,
  landform: 'plain' | 'valley' | 'ridge' | 'coast' | 'forest' | 'mesa' | 'wetland',
  enclosure,
  vistaSide,
  routeClass,
  landmark,
  weatherBias,
  paletteScript,
  densityProfile
}
```

Use a constrained state machine or weighted grammar rather than independent random choices. Example:

```text
rest -> tease -> transition -> reveal -> landmark -> cooldown -> rest
                 \-> false reveal -> transition -/
```

Rules should prevent repetition and implausible adjacency:

- no more than two enclosed beats in a row;
- a major landmark requires a minimum cooldown distance;
- coast beats keep water on a stable side for several chunks;
- a tunnel requires a mountain/ridge approach;
- a bridge requires a river, ravine, marsh, or coast causeway;
- biome changes take hundreds of meters unless a deliberate threshold event explains them;
- severe curve, steep grade, and dense clutter do not peak simultaneously;
- the most dramatic vista appears near a road crest or bend reveal.

This is the single highest-value system for making procedural output feel authored.

### 3.3 Use nested distance scales

Generate variation at distinct scales so systems do not all change together:

| Scale | Distance | Controls |
|---|---:|---|
| Journey | 5-20 km | broad biome arc, climate, time progression |
| Chapter | 1-4 km | landform family, route class, major weather |
| Scenic beat | 250-800 m | enclosure, vista, landmark, reveal |
| Chunk | 48-128 m | road mesh, terrain patch, object batches |
| Detail cell | 4-16 m | grass, rocks, decals, particles |

Use independent hashed streams for each scale and category. Adding a flower must not move a tree or change the road. Named random domains such as `route`, `landform`, `landmark`, `flora`, `weather`, and `microDetail` prevent this form of procedural butterfly effect.

### 3.4 Chunk the route and cache materialized results

Use road-distance chunks with a ring buffer around the car:

- `critical`: current and next 1-2 chunks, collision-ready;
- `near`: full road and terrain, high-detail instances, shadows;
- `mid`: simplified terrain, reduced flora, no small props;
- `far`: silhouette terrain, tree clusters/impostors, landmarks;
- `horizon`: skyline cards or a very low-frequency terrain ring.

Build each chunk once when it enters the active window. Shift/reuse buffers as the car advances. Limit generation to a fixed time slice per frame, with emergency prefetch based on speed. Chunk descriptors should be pure data and transferable to a worker later.

This follows the same useful idea as geometry clipmaps: nested regular regions around the viewer, incrementally updated as the viewpoint moves, giving predictable work and stable rendering cost.

### 3.5 Preserve two useful forms of determinism

- **State determinism:** mandatory across supported hardware for oracle trajectory and semantic world plan.
- **Pixel determinism:** required only on the pinned data-generation renderer/configuration.

Do not require decorative shader wind or GPU particles to be bit-identical across all devices. For training export, drive their phase from integer simulation step and use the fixed renderer profile.

---

## 4. Road generation: make the route the hero

### 4.1 Replace raw curvature noise with a route profile

Generate a target curvature profile in semantic segments:

- straight;
- gentle arc;
- S-curve;
- switchback;
- crest approach;
- descent;
- vista bend;
- bridge/tunnel approach.

Blend segments with curvature-continuous transitions. A practical implementation does not need a full civil-engineering solver: use piecewise cubic curvature ramps or clothoid approximations, constrain curvature derivative, and integrate heading from the smoothed profile.

Constraints:

- minimum radius by route class and target speed;
- maximum curvature change per meter;
- no blind sharp turn immediately after a crest;
- minimum recovery straight after a demanding sequence;
- avoid local self-intersection and reject chunks that collide with recent route history;
- reserve outside space for a vista on reveal bends.

### 4.2 Generate a road grade profile independently from raw terrain

The current centerline copies terrain height. Instead:

1. Sample desired terrain along a look-ahead route.
2. Fit a smooth grade profile under maximum slope and vertical-curvature limits.
3. Classify the difference between road and terrain as cut, fill, causeway, bridge, or tunnel.
4. Modify nearby presentation terrain to meet the road with biome-appropriate shoulders.

This creates plausible engineering and gives scenery meaningful forms: embankments, exposed cuts, retaining walls, viaducts, and overlooks.

### 4.3 Add route classes

Route class is a slowly varying descriptor, not necessarily a user-facing dial:

- country lane: narrow, no center line in places, hedges/fences, low speeds;
- scenic two-lane: center and edge markings, overlooks, guardrails;
- mountain road: switchbacks, rock cuts, snow poles, barriers;
- coastal road: open water side, drainage, sea walls, bridges;
- forest track: broken edges, canopy enclosure, occasional gravel;
- surreal causeway: clean graphic geometry, crystals/monoliths, dramatic sky.

Each class controls width, crown/camber, markings, shoulder, prop grammar, curve limits, and expected speed. Keep classes smoothly blendable where they are used in training data.

### 4.4 Add camber and road cross-section

Represent the road as a small cross-section swept along the centerline:

- pavement crown;
- shoulder;
- ditch or curb;
- optional barrier sockets;
- bank angle based on signed curvature and route class.

Banking should be physically present in the oracle if it affects traction. Cosmetic shoulder details can stay in presentation data.

### 4.5 Improve road visual language

- Make edge width stable in screen space at distance.
- Use markings as geometry or analytic shader masks, not fragile thin texture detail.
- Add subtle broad albedo variation keyed by chunk, not noisy asphalt texture.
- Add wet response as lower roughness and darker value, with restrained sky reflection.
- Place skid marks, repaired patches, leaves, snow edges, and puddles as sparse semantic decals.
- Use roadside delineators before turns and reflectors at night to communicate curvature.

---

## 5. Terrain and scenery

### 5.1 Replace the single road ribbon with a near/mid/far terrain system

The current 96 x 65 road-oriented strip is rebuilt every frame. Move toward nested terrain bands:

- near road-conforming mesh for exact road/shoulder contact;
- mid terrain tiles or clipmap rings with moderate resolution;
- far low-frequency silhouette ring;
- horizon landmarks rendered independently so a mountain does not need dense connecting geometry.

Morph or dither between LODs over distance. Hide transitions in fog and avoid geometry popping in exported clips.

### 5.2 Use landform grammars, not just noise octaves

Noise should perturb authored forms, not define the entire world. Build composable signed or height-field features:

- ridge chains with a preferred direction;
- valleys and drainage corridors;
- plateaus and stepped mesas;
- coastal shelf plus sea stacks;
- rolling moraine hills;
- basin/lake systems;
- dune fields;
- terraced surreal islands;
- volcanic cones or crystal ridges for alien space.

The scenic director chooses and orients these forms relative to the route. Add low-amplitude noise last for natural irregularity.

### 5.3 Establish three depth layers

Every scenic beat should attempt to populate:

- **Foreground:** shoulder grass, posts, fences, road cuts, occasional overhanging branches. Strong motion parallax and surface speed cues.
- **Midground:** tree groups, buildings, boulders, rivers, fields. Defines the current place.
- **Background:** ridges, mountains, mesa silhouettes, cloud banks, distant structures. Frames the horizon and destination.

Budget density separately per layer. The current scene often has dense midground and an empty background; adding more near trees will not solve that.

### 5.4 Group vegetation ecologically

Uniform scatter reads synthetic even when each point is random. Use clusters and exclusion fields:

- parent groves with child trees;
- treelines following water or elevation bands;
- sparse exposed ridges;
- denser sheltered valleys;
- species succession across biome transitions;
- clearings around viewpoints and landmarks;
- asymmetric roadside growth;
- fallen logs and shrubs near forest edges;
- no grass underwater, on asphalt, or on steep exposed rock.

Store cluster descriptors per chunk; derive individual instances from the descriptor. This improves coherence and makes far LOD easy because a grove already has a cluster representation.

### 5.5 Build small authored prop kits

Use 8-15 strong pieces per biome rather than hundreds of weak assets. Geometry can remain generated or tiny glTF assets:

- 3-4 tree silhouettes at different ages;
- 2 trunk/canopy constructions;
- 3 rock families;
- one ground-detail family;
- one roadside infrastructure family;
- one landmark family;
- one settlement/structure family.

Allow deterministic scale, mirror, lean, palette, and modular composition. Preserve chunky proportions and avoid thin geometry that aliases at training resolution.

### 5.6 Introduce landmarks and micro-events

Major landmarks, spaced roughly 1-4 km apart:

- natural arch;
- waterfall crossing;
- lone enormous tree;
- castle/temple silhouette;
- observatory;
- wind farm;
- giant crystal formation;
- viaduct;
- tunnel portal;
- cliff overlook;
- village skyline;
- aurora basin.

Micro-events, spaced more frequently but with cooldowns:

- flock crossing the sky;
- leaves or snow blown across the road;
- a parked camper at an overlook;
- short covered bridge;
- grazing silhouettes;
- roadside shrine/sign;
- construction cones;
- puddle splash;
- fireflies at dusk;
- brief sun shaft as trees open.

Events should be annotations in the world plan so they can be labeled, excluded, or balanced in datasets.

### 5.7 Add optional forks carefully

Forks create agency and greatly increase replay appeal, but they complicate nearest-road queries, skeleton representation, and training. Treat them as a later experiment:

- first ship scenic pullouts and apparent side roads that do not change the route;
- then add rare binary forks with both branches generated ahead;
- encode route graph and selected branch in snapshots and anchor channels;
- avoid intersections until the world model reliably handles a single ribbon.

---

## 6. Art direction: stylized, iconic, compressible

### 6.1 Define shape language by biome

Color is secondary. Each biome needs dominant geometric motifs:

| Biome family | Dominant shapes | Surface rhythm | Skyline |
|---|---|---|---|
| Temperate | rounded crowns, tapered trunks | clustered fields/forest | rolling layered ridges |
| Alpine | triangular conifers, angular rock | vertical repetition | sharp peaks and passes |
| Arid | slabs, columns, low scrub | open negative space | mesas and distant stacks |
| Coastal | windswept asymmetry, low brush | horizontal bands | headlands and islands |
| Snow | softened caps, dark vertical trees | broad quiet areas | high-contrast ridges |
| Alien | fins, crystals, impossible arches | sparse graphic clusters | stepped or tilted silhouettes |

At 128px, a scene should remain classifiable by silhouette and value grouping.

### 6.2 Use deliberate palette scripts

Each scenic chapter gets a compact palette:

- sky zenith;
- horizon haze;
- sun/key light;
- ground light/mid/dark;
- foliage light/mid/dark;
- rock;
- road;
- one accent.

Generate colors from artist-authored ramps, not arbitrary HSL shifts. Keep the road separated from the ground by value at all times. Use atmospheric perspective to compress distant values toward the horizon color.

Biome dials can blend between palette endpoints continuously, but the endpoints need art direction. Add palette regression screenshots for extreme dial corners.

### 6.3 Improve materials without adding noisy textures

- Use flat or toon lighting with controlled face-value steps.
- Add baked/analytic vertex ambient occlusion at prop intersections and terrain creases.
- Add a simple foliage transmission term when backlit.
- Use broad triplanar color bands for rock/soil only if they remain stable at 128px.
- Give wet roads and water coherent specular shapes, not full-screen bloom.
- Reserve emissive accents for night readability and alien landmarks.
- Consider a restrained depth/normal edge accent only on hero objects; full-scene outlines can shimmer and burden the model.

### 6.4 Make the sky a compositional layer

- Add cloud families: high streak, fair-weather stack, storm shelf, valley fog bank.
- Arrange clouds around the vista direction rather than evenly around the camera.
- Ensure cloud motion has multiple parallax depths.
- Give distant rain a visible curtain before local precipitation begins.
- Add controlled sunrise/sunset color scripts and moonlight, with exposure adapted slowly.
- Treat aurora as broad stable ribbons, not high-frequency noise.

The current sprite clouds orbit the camera in one ring. A layered weather volume will produce much stronger depth at similar cost.

### 6.5 Use fog to reveal depth, not erase it

Keep exponential haze, but coordinate it with distance layers:

- near geometry retains contrast;
- midground shifts toward haze color;
- background becomes a controlled silhouette;
- the road remains readable through stronger local contrast/reflectors;
- heavy fog reduces generated far-detail budget because it cannot contribute.

Fog is both an art tool and a streaming mask. Density changes should be slow enough that hidden chunks are ready before fog clears.

---

## 7. Camera and driving feel

### 7.1 Fix frame-rate dependence first

The demo currently calls one fixed simulation step per `requestAnimationFrame` and smooths input by a fixed amount per displayed frame. On a 120 Hz display the simulation advances twice as fast as on a 60 Hz display.

Use a real-time accumulator:

- sample input immediately every display frame;
- run zero or more fixed 1/60 or 1/30 simulation steps from accumulated wall time;
- cap catch-up steps to avoid a death spiral;
- render an interpolation of previous/current physical state;
- smooth input with a time-based exponential or in fixed simulation steps;
- keep direct input sampling and camera presentation at display rate.

This preserves deterministic physics while reducing visible judder and keeping controls consistent across refresh rates.

### 7.2 Build a deterministic spring-arm camera

Desired behavior:

- lower car occupancy: roughly 12-20% of frame height, not a dominant block;
- look ahead along the road centerline, not only along car heading;
- shift the aim point slightly toward the inside of upcoming curves;
- keep the horizon in a stable compositional band;
- damp position more than aim direction;
- add small acceleration pitch and lateral load roll;
- filter terrain pitch so every road sample does not shake the camera;
- increase distance/FOV gently with speed, using one or the other primarily to avoid exaggerated zoom;
- correct clipping with a cheap terrain/obstacle probe;
- offer close chase, scenic chase, hood, and optional auto-drive camera profiles.

Camera state used for training renders must be replayable. Either store it in snapshots or define it as a deterministic filter advanced once per simulation step.

### 7.3 Add visual suspension without full rigid-body cost

Keep the simple physical oracle, then add a deterministic four-contact visual rig:

- sample road/terrain under each wheel;
- spring wheel meshes toward contact height;
- derive body heave, pitch, and roll from filtered wheel contacts;
- add acceleration squat, brake dive, and cornering roll within small limits;
- add a short landing compression impulse;
- keep the collision body simple.

This gives the car weight while avoiding a heavyweight physics engine and unstable data.

### 7.4 Improve the compact vehicle model

Recommended additions, in order:

1. Speed-sensitive steering curve and steering return.
2. Yaw velocity as explicit state rather than heading changing instantly from input.
3. Lateral velocity/grip saturation with a stable tire-force approximation.
4. Longitudinal load effect on available cornering grip.
5. Bank and slope contribution.
6. Distinct braking vs reversing input behavior.
7. Mild driving assists for the calm default: counter-steer damping and off-road recovery.

The goal is controllable weight, not simulation-grade tire telemetry. Keep all equations deterministic, bounded, and testable at dial extremes.

### 7.5 Communicate speed and surface

Use several subtle cues instead of aggressive motion blur:

- roadside posts and near grass with strong parallax;
- wheel rotation and suspension movement;
- wind and tire audio increasing with speed;
- small dust, spray, snow, or leaf wake based on surface;
- camera vibration only on rough surfaces, low amplitude and band-limited;
- narrow FOV change;
- passing shadow patterns under trees;
- road marking cadence;
- optional speed lines only for surreal/high-speed modes.

Motion blur is a poor fit for low-entropy training frames and can hide latency. Avoid it in the oracle renderer.

---

## 8. Sound and ambient simulation

Even though it is not graphics, sound is one of the cheapest ways to make the simulation feel embodied.

Use Web Audio with a small procedural mix:

- engine tone from speed, throttle, and load;
- tire/rolling layer by surface and slip;
- wind layer by speed and openness;
- suspension/impact one-shots;
- rain on body, thunder, snow hush;
- biome ambience: birds, insects, coast, forest wind, distant settlement;
- landmark stingers used sparingly;
- smooth interior/exterior filtering by camera mode.

Audio events must be keyed to simulation state. Ambient one-shots can be deterministically scheduled by distance chunk.

---

## 9. Performance and latency architecture

### 9.1 Define budgets before adding content

Standalone simulator target on a representative laptop:

| Budget | 60 Hz target | 30 Hz beside model |
|---|---:|---:|
| Input + fixed-step core | <= 0.5 ms CPU | <= 0.5 ms CPU |
| Chunk scheduling/materialization | <= 1.0 ms amortized CPU | <= 0.5 ms amortized CPU |
| Render submission | <= 2.0 ms CPU | <= 1.5 ms CPU |
| GPU scene + post | <= 8.0 ms GPU | <= 4-6 ms GPU |
| Long-frame p99 | < 20 ms | < 40 ms total product frame |

These are starting gates, not promises. Measure on integrated GPU, discrete GPU, and a low-power machine. The final M4 budget must be set after tokenizer/dynamics/decoder profiling.

### 9.2 Eliminate hot-path regeneration

- Build terrain and scatter only when chunks enter or their semantic descriptor changes.
- Reuse typed arrays, geometries, object records, and worker messages.
- Use object pools for chunks and transient effects.
- Update only changed instance ranges.
- Compute static normals and bounds once per chunk.
- Separate dial changes that require geometry regeneration from shader-only presentation changes.
- Quantize slow dial changes for expensive resources while preserving smooth shading transitions.

### 9.3 Use instancing by LOD and material

Keep `InstancedMesh`, but split instances into spatial batches with valid bounds so frustum culling works. For each prop family:

- near: full low-poly mesh, receives/casts selected shadows;
- mid: reduced mesh, no shadow casting;
- far: cluster mesh, point sprite, or alpha-tested impostor;
- beyond fog contribution: omitted.

Give important hero trees and landmarks separate batches. One global instance batch with `frustumCulled = false` makes every view pay for every active object.

### 9.4 Make shadows selective and temporally cached

- Cast dynamic shadows from the car and near hero vegetation only.
- Use blob/contact shadow under the car for guaranteed grounding.
- Bake or analytically approximate AO for static props.
- Tighten the shadow camera around the visible near road.
- Lower shadow resolution or update frequency by quality tier.
- Freeze shadow maps when sun, car cell, and caster set have not changed enough.
- Disable small grass/bush shadow casting.

At low-poly fidelity, coherent contact and directional value separation matter more than a 2048 map covering hundreds of trees.

### 9.5 Move ambient motion to shaders

Tree sway, grass wind, water ripple, and much precipitation motion can derive from world position, instance phase, and integer simulation time in vertex/fragment shaders. This preserves draw-call efficiency and removes thousands of CPU updates.

Keep collision geometry static. Wind is presentation unless a future research experiment explicitly makes it physical.

### 9.6 Use quality tiers and dynamic resolution

Create explicit profiles, not scattered conditionals:

- **Data:** fixed 128/256 capture, deterministic effects, no unnecessary display post.
- **Low:** reduced far detail, 1x shadows, no bloom, lower DPR.
- **Balanced:** near shadows, far clusters, restrained post.
- **High:** denser layers, better water/clouds, high DPR.
- **Anchor runtime:** skeleton only, no decorative simulation.

Adjust render scale from smoothed GPU frame time with hysteresis. Do not change physical simulation quality or oracle state by tier.

### 9.7 Avoid post-processing as the main look

The current half-float 4x MSAA composer plus bloom is expensive relative to the stylized scene. Make the base render attractive without post. Then:

- combine color grade, vignette, and optional posterization in one pass;
- use bloom only for emissive/night highlights and allow it to disable cleanly;
- consider lower-resolution bloom;
- avoid TAA unless temporal stability is proven for exported data;
- expose render-scale and post costs in the debug HUD.

### 9.8 Prepare worker boundaries

Good worker candidates:

- future chunk world-plan generation;
- terrain/road mesh attribute generation;
- scatter/cluster materialization;
- exporter encoding and I/O.

Keep immediate input, physical stepping, camera, and render submission on the main thread unless measured contention justifies moving them. Workers reduce spikes but add transfer and synchronization costs; transfer typed buffers, do not clone object graphs.

---

## 10. World-model and dataset constraints

The beautiful simulator and useful oracle must remain the same project.

### 10.1 Split factors into three categories

1. **Steering factors:** deliberately continuous and independently sampled, such as fog, precipitation, time, broad biome weights, and selected physics values.
2. **Labeled nuisance factors:** route class, landform, prop kit, landmark type, cloud family. These enrich data but are not initially exposed as activation directions.
3. **Presentation-only factors:** quality tier, display resolution, non-semantic LOD transitions. These must not leak unpredictably into training captures.

Do not turn every art parameter into a product dial. Too many independent axes increase dataset requirements and make feature disentanglement harder.

### 10.2 Label the world plan

Export per-frame or per-chunk metadata:

- scenic beat ID and role;
- biome weights and chapter ID;
- route class, curvature, grade, bank;
- landform and enclosure;
- landmark/event labels and distance;
- weather front state;
- semantic visible-object counts;
- camera profile;
- quality/capture profile.

This enables stratified datasets, hard-case evaluation, and later interpretability analysis.

### 10.3 Expand auxiliary heads

In addition to depth and segmentation, consider cheap oracle channels:

- road tangent/curvature look-ahead vector;
- drivable mask;
- optical flow from known camera/object transforms;
- surface normals;
- instance IDs for landmarks/obstacles;
- horizon/sky mask;
- wheel-contact and grounded state;
- coarse scene graph tokens for the lambda anchor.

Do not commit all channels to model conditioning. Generate them because the sim makes them cheap; ablate later.

### 10.4 Protect temporal stability

Reject or gate features that produce unstable subpixel detail:

- thin grass lines at distance;
- hard LOD pops;
- noisy screen-space dithering that changes every frame;
- uncontrolled alpha-blended particles;
- rapidly adapting exposure;
- shadow-map shimmer;
- reflective detail smaller than a model pixel.

Run temporal-difference heatmaps on static-camera and constant-action sequences. A still world should not sparkle.

### 10.5 Keep contrastive pairs truly matched

Changing a steering dial must not reseed layout or reorder unrelated random streams. If fog changes, the exact same road, props, events, and camera path must remain underneath it. This requires named random domains and chunk IDs independent of dial values.

For geometry-changing dials such as hilliness, define whether contrastive pairs intentionally change geometry. Do not accidentally conflate a biome direction with an entirely different road.

---

## 11. Instrumentation and evaluation

### 11.1 Add a simulator diagnostics overlay

Toggleable, absent from captures:

- FPS and CPU/GPU frame-time graph;
- fixed steps this frame and accumulator lag;
- draw calls, triangles, points, and active instances;
- chunk queue, generation time, cache hits, and memory;
- shadow/post costs;
- current chapter, beat, route class, biome, and landmark;
- camera target and road look-ahead;
- input-to-present event markers where browser APIs permit.

Use `renderer.info` plus CPU performance marks initially; add disjoint timer queries where available for GPU timings.

### 11.2 Build deterministic visual regression routes

Create 8-12 golden journeys, each defined by seed, dial schedule, and action script:

- default temperate day;
- alpine snow;
- arid sunset;
- coastal fog;
- forest rain;
- alien night/aurora;
- maximum curvature/hilliness stress;
- low gravity/friction stress;
- off-road collision sequence;
- long 10 km streaming run.

Capture fixed-distance frames, not just fixed wall-clock times. Compare pixels, segmentation, state hashes, and performance traces.

### 11.3 Add experience metrics

Not everything valuable is a render metric. Track:

- road visible area and visible look-ahead distance;
- car screen occupancy;
- horizon height and background silhouette occupancy;
- time since last landmark/event;
- repetition distance for prop/beat sequences;
- curve difficulty envelope and recovery time;
- steering reversal rate and off-road time in scripted/autopilot runs;
- frame-time p50/p95/p99 and worst chunk-generation spike;
- temporal flicker score on static content;
- entropy/compressibility of 128px training captures.

Use these as alarms, not as a substitute for visual review.

### 11.4 Add an autopilot for testing and relaxed play

A deterministic road follower can:

- produce stable dataset action distributions;
- run unattended streaming soak tests;
- make visual regression captures repeatable;
- let players use the simulator as a scenic screen experience;
- expose route difficulty failures by logging required steering and error.

Use look-ahead curvature and lateral error with a simple pure-pursuit or Stanley-style controller. Record actions exactly like human inputs.

---

## 12. Implementation roadmap

### Phase 0: Baseline and guardrails

Goal: know the cost and visual failure modes before architectural changes.

- Add fixed-distance screenshot scripts for representative seeds/dials.
- Add CPU frame marks, `renderer.info`, active instance counts, and chunkless baseline traces.
- Record p50/p95/p99 frame times on three hardware classes if available.
- Add tests for refresh-rate-independent simulation advancement.
- Add determinism hashes for state, world plan, and snapshots.
- Document current capture color pipeline; display and data capture currently differ because `capture()` bypasses post-processing.

Exit criteria: reproducible visual/performance baseline and explicit budgets.

### Phase 1: Driving and camera vertical slice

Goal: make the existing world immediately feel better before expanding content.

- Implement fixed-timestep accumulator plus render interpolation.
- Make input smoothing time-based.
- Build road-curvature look-ahead camera with deterministic spring state.
- Reduce car occupancy and expose more immediate road.
- Add visual suspension and surface particles.
- Add engine/tire/wind audio prototype.
- Tune steering curve, yaw state, and recovery assists.

Exit criteria: default seed is pleasant for a five-minute drive at 60 and 120 Hz; scripted control produces the same oracle trajectory at both refresh rates.

### Phase 2: World plan and scenic director

Goal: replace uniform randomness with authored procedural rhythm.

- Define `JourneyPlan`, `Chapter`, `ScenicBeat`, and named random domains.
- Implement deterministic beat grammar with repetition constraints.
- Connect existing landmark candidates to beat decisions.
- Add route classes and semantic road segments.
- Implement 3-4 landform families and background silhouettes.
- Add a debug timeline showing upcoming beats.

Exit criteria: a 10 km journey has recognizable chapters, at least three reveal patterns, controlled cooldowns, and exact replay.

### Phase 3: Streaming renderer

Goal: add diversity while lowering hot-path cost.

- Introduce 64-128 m chunk descriptors and active ring buffer.
- Cache road/terrain geometry and scatter instances per chunk.
- Replace per-frame terrain rebuild with near/mid/far bands.
- Spatially batch instancing and restore frustum culling.
- Add prop LODs/cluster impostors.
- Time-slice chunk generation; move mesh/scatter materialization to a worker if profiling supports it.
- Add adaptive detail and render-scale quality profiles.

Exit criteria: no visible pop or >4 ms CPU generation spike during a 10 km soak; scene CPU cost decreases despite greater variety.

### Phase 4: Art-kit pass

Goal: make every chapter visually identifiable and memorable.

- Build temperate, alpine, arid, coastal, and alien shape/palette kits.
- Add ecological clustering and asymmetric roadside composition.
- Add cut/fill/bridge/tunnel road treatments.
- Add one major landmark and 3-5 micro-events per kit.
- Add cloud/weather families and layered sky composition.
- Add restrained material upgrades: AO, foliage transmission, wetness, snow accumulation.

Exit criteria: blind 128px frames are distinguishable by biome and scenic role; repeated primitives are not obvious in a five-minute capture.

### Phase 5: Dataset and anchor integration

Goal: ensure the richer sim strengthens the research pipeline.

- Export world-plan labels and expanded aux channels.
- Define canonical data render profile.
- Validate matched contrastive pairs after architecture changes.
- Measure frame entropy, tokenizer reconstruction, and temporal flicker by feature.
- Implement skeleton token schema for route look-ahead, terrain, and semantic events.
- Run ablations: base sim vs camera upgrade vs scenic director vs full art kit.

Exit criteria: richer data improves perceived quality without unacceptable tokenizer cost or reduced steering disentanglement.

### Phase 6: Polish and product modes

- Add auto-drive, photo/view mode, and a minimal product HUD separate from the debug panel.
- Add rare forks only if single-route model stability is strong.
- Tune audio mix, weather transitions, night readability, and accessibility.
- Add graceful WebGL/WebGPU capability tiers and warmup.
- Lock a benchmark journey for M4 on-device demonstrations.

---

## 13. Recommended first five PRs

### PR 1: Frame loop correctness and instrumentation

- accumulator, interpolation, time-based input;
- frame-time HUD and `renderer.info`;
- refresh-rate determinism test;
- scripted browser capture for baseline seeds.

This fixes a correctness issue and makes every later optimization measurable.

### PR 2: Scenic chase camera and visual suspension

- curve-aware aim target;
- deterministic position/rotation springs;
- car framing profiles;
- filtered wheel contacts and body response;
- camera regression captures.

This should deliver the largest immediate improvement in perceived quality.

### PR 3: Chunk data model and cache

- pure chunk IDs/descriptors;
- named random streams;
- active distance ring;
- cached scatter and terrain arrays;
- no visible art change initially.

This is the enabling performance architecture.

### PR 4: Scenic director vertical slice

- one 3 km chapter sequence;
- rest/tease/reveal/cooldown roles;
- valley, ridge, and coast landforms;
- one landmark type;
- debug beat timeline.

This tests whether authored rhythm solves the core experience problem before producing many assets.

### PR 5: Two complete art kits

- temperate and alpine shape languages;
- palette scripts;
- ecological clustering;
- near/mid/far representations;
- one weather family and several micro-events each.

Two polished kits are enough to validate variety, transitions, model compressibility, and production cost.

---

## 14. Prioritized feature backlog

| Priority | Feature | Experience value | Performance risk | Research risk |
|---|---|---:|---:|---:|
| P0 | Fixed accumulator + interpolation | High | Low | Low |
| P0 | Curvature-aware camera/framing | Very high | Low | Medium: camera must replay |
| P0 | Profiling + golden journeys | High | Low | Low |
| P0 | Chunk caching and named random streams | High | Low | Low |
| P0 | Scenic beat director | Very high | Low | Medium: more labels |
| P1 | Route profiles and grade constraints | Very high | Medium | Medium: changes oracle |
| P1 | Near/mid/far terrain | High | Medium | Low |
| P1 | Biome shape/palette kits | Very high | Medium | Medium: dataset entropy |
| P1 | Vegetation clusters + LOD | High | Low | Low |
| P1 | Visual suspension | High | Low | Low if presentation-only |
| P1 | Sound and surface feedback | High | Low | None |
| P1 | Landmarks and micro-events | Very high | Medium | Medium: balance data |
| P2 | Bridges, tunnels, cuts, retaining walls | High | Medium | Medium |
| P2 | Shader wind and weather fronts | Medium | Low | Medium: temporal stability |
| P2 | Selective cached shadows | Medium | Low | Low |
| P2 | Autopilot | High | Low | Positive for data |
| P2 | Dynamic resolution/quality tiers | Medium | Medium | Low if captures fixed |
| P3 | Rare route forks | High | High | High |
| P3 | Traffic, pedestrians, wildlife agents | Medium | High | High |
| P3 | Destructible props | Low for thesis | High | High |

---

## 15. Ideas to reject or defer

- **Photoreal textures and dense scan assets:** increase entropy, download size, material count, and tokenizer burden without solving composition.
- **Full rigid-body vehicle physics immediately:** costly to tune and can reduce deterministic stability. Improve the compact model first.
- **More independent user dials for every art choice:** harms coverage and steering disentanglement. Use labeled nuisance factors.
- **Random landmarks everywhere:** destroys scenic rhythm.
- **Per-frame procedural rebuilding:** scales cost with ambition and causes spikes.
- **Heavy motion blur, depth of field, film grain, and unstable outlines:** poor for latency perception and autoregressive learning.
- **A giant asset library before the director exists:** repetition is primarily structural today.
- **WebGPU migration as an art milestone:** it may become useful for M4, but architecture and profiling should identify the actual bottleneck first.
- **Forks/intersections before single-road quality:** they multiply model and oracle complexity.

---

## 16. Success definition

The simulator upgrade is successful when:

1. A player can drive for ten minutes and describe multiple distinct places and moments, not only colors or weather.
2. The road remains readable through curves, crests, fog, and night.
3. The same seed and actions reproduce physical state, world plan, semantic events, and canonical training frames.
4. Frame-time p99 remains within budget during chunk transitions and weather/biome changes.
5. The skeleton/anchor remains a small deterministic data structure rather than inheriting render complexity.
6. 128px captures retain clear silhouettes, stable temporal edges, and manageable entropy.
7. Contrastive dial pairs remain matched except for the intended factor.
8. The richer simulator produces measurable gains in tokenizer reconstruction quality, rollout coherence, or human preference without breaking on-device targets.

The experiential bar is simple: it should no longer feel like a procedural test scene with a car in it. It should feel like a road trip with a quiet sense of authorship, even though the journey has no end.

---

## 17. Research references

These sources informed the architecture and recommendations:

- Glenn Fiedler, [Fix Your Timestep!](https://gafferongames.com/post/fix_your_timestep/) — fixed simulation steps, render interpolation, catch-up limits, and reproducibility.
- Losasso and Hoppe / NVIDIA, [Terrain Rendering Using GPU-Based Geometry Clipmaps](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry) — nested viewer-centered grids, incremental updates, stable rendering rate, and graceful LOD.
- Ryan Geiss / NVIDIA, [Generating Complex Procedural Terrains Using the GPU](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu) — block pools, visible-region priority, reuse, and procedural terrain generation.
- David Whatley / NVIDIA, [Toward Photorealism in Virtual Botany](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-1-toward-photorealism-virtual-botany) — deterministic planting, spatial cells, and layered vegetation management. The rendering goal here is stylized, but the scene-management lessons transfer.
- Renaldas Zioma / NVIDIA, [GPU-Generated Procedural Wind Animations for Trees](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-6-gpu-generated-procedural-wind-animations-trees) — cheap phenomenological wind, vertex-shader animation, and instancing.
- Eric Risser / NVIDIA, [True Impostors](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-21-true-impostors) — image-based representations for dense distant objects and their tradeoffs.
- Georgios Yannakakis and Julian Togelius, [Experience-Driven Procedural Content Generation](https://yannakakis.net/wp-content/uploads/2019/02/EDPCG.pdf) — treating generated content as a means to shape player experience rather than maximizing raw novelty.
- Chen et al., [Interactive Procedural Street Modeling](https://www2.cs.uh.edu/~chengu/Publications/streetModeling/street_modeling.html) — combining global fields, local constraints, smoothing, and editable procedural road structure.
- Three.js, [documentation](https://threejs.org/docs/) — current instancing, LOD, renderer information, fog, shadow, and node-material capabilities.

