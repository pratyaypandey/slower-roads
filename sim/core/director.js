// Scenic director — deterministic authored rhythm over road distance (SIM_UPGRADE §3.2).
//
// Instead of one continuously-randomized surface, the drive is a sequence of *scenic
// beats* (250–800 m) grouped into *chapters* (1–4 km). A constrained grammar chooses
// each beat's role, landform, enclosure, vista side, route class, and (rarely) a
// landmark, with rules that prevent repetition and implausible adjacency. Everything is
// a pure function of (seed, road distance), keyed on arc-length, so the same seed always
// produces the same journey and replays exactly (SIM_UPGRADE §3.3 named random domains).
//
// This is world *plan*, not presentation: the renderer materializes it, the physics and
// dataset read its labels. It never overrides the player's steering dials (fog, biome);
// it supplies labeled nuisance structure (SIM_UPGRADE §10.1).

import { hash1 } from './prng.js';

// Independent hashed streams — adding a landmark must not move a landform or the route.
const DOMAIN = { chapter: 101, beat: 211, landform: 307, route: 401, landmark: 509, vista: 601, biome: 701 };

export const ROLES = ['rest', 'tease', 'transition', 'reveal', 'landmark', 'cooldown'];
export const LANDFORMS = ['plain', 'valley', 'ridge', 'coast', 'forest', 'mesa', 'wetland'];
export const ROUTE_CLASSES = ['country', 'scenic', 'mountain', 'coastal', 'forest', 'causeway'];
export const LANDMARKS = ['arch', 'lone_tree', 'monolith_ring', 'crystal_spire', 'tower', 'viaduct'];

// How enclosed each landform feels (0 open vista .. 1 dense enclosure). Drives scatter
// density and the reveal/rest contrast.
const ENCLOSURE = { plain: 0.18, valley: 0.8, ridge: 0.32, coast: 0.1, forest: 1.0, mesa: 0.16, wetland: 0.62 };

// Weighted next-role transitions: rest -> tease -> transition -> reveal -> landmark ->
// cooldown -> rest, with false-reveal and rest branches so it never feels mechanical.
const NEXT_ROLE = {
  rest: ['tease', 'tease', 'transition', 'rest'],
  tease: ['transition', 'reveal', 'transition'],
  transition: ['reveal', 'landmark', 'rest'],
  reveal: ['landmark', 'cooldown', 'transition'],
  landmark: ['cooldown', 'cooldown', 'rest'],
  cooldown: ['rest', 'rest', 'tease'],
};

const LANDMARK_COOLDOWN = 900;   // min metres between placed landmarks
const OPEN_LANDFORMS = ['plain', 'coast', 'mesa', 'ridge'];

const pick = (arr, h) => arr[Math.min(arr.length - 1, Math.floor(h * arr.length))];

export class Director {
  constructor(seed) {
    this.seed = seed >>> 0;
    this._chapters = [];
    this._beats = [];
    this._lastLandmarkD = -Infinity;
  }

  _h(domain, i, salt = 0) {
    return hash1((Math.imul(i, 2654435761) ^ Math.imul(domain, 40503) ^ Math.imul(salt + 1, 2246822519)) >>> 0, this.seed);
  }

  // --- chapters -------------------------------------------------------------
  _ensureChapters(d) {
    if (this._chapters.length === 0) this._chapters.push(this._buildChapter(0, 0));
    let last = this._chapters[this._chapters.length - 1];
    while (last.d1 < d) { last = this._buildChapter(last.index + 1, last.d1); this._chapters.push(last); }
  }

  _buildChapter(index, d0) {
    const len = 1000 + this._h(DOMAIN.chapter, index) * 3000;
    return {
      index, d0, d1: d0 + len,
      routeClass: pick(ROUTE_CLASSES, this._h(DOMAIN.route, index)),
      biome: { x: this._h(DOMAIN.biome, index, 1), y: this._h(DOMAIN.biome, index, 2) },
    };
  }

  chapterAt(d) {
    this._ensureChapters(d);
    for (let i = this._chapters.length - 1; i >= 0; i--) if (d >= this._chapters[i].d0) return this._chapters[i];
    return this._chapters[0];
  }

  // --- beats ----------------------------------------------------------------
  _ensureBeats(d) {
    if (this._beats.length === 0) this._beats.push(this._buildBeat(0, 0));
    let last = this._beats[this._beats.length - 1];
    while (last.d1 < d) { last = this._buildBeat(last.index + 1, last.d1); this._beats.push(last); }
  }

  _buildBeat(index, d0) {
    const prev = index > 0 ? this._beats[index - 1] : null;
    const prev2 = index > 1 ? this._beats[index - 2] : null;
    const chapter = this.chapterAt(d0);

    const role = pick(NEXT_ROLE[prev ? prev.role : 'rest'], this._h(DOMAIN.beat, index));

    // Landform, with the "no more than two enclosed in a row" constraint.
    let landform = pick(LANDFORMS, this._h(DOMAIN.landform, index));
    if (prev && prev2 && ENCLOSURE[prev.landform] > 0.6 && ENCLOSURE[prev2.landform] > 0.6) {
      landform = pick(OPEN_LANDFORMS, this._h(DOMAIN.landform, index, 5));
    }
    const enclosure = ENCLOSURE[landform];

    // Vista side stays stable across a run of coast beats (water on one side).
    let vistaSide = prev ? prev.vistaSide : 1;
    if (!(landform === 'coast' && prev && prev.landform === 'coast')) {
      vistaSide = this._h(DOMAIN.vista, index) < 0.5 ? -1 : 1;
    }

    // Landmarks only on landmark-role beats, and only after a cooldown distance.
    let landmark = null;
    if (role === 'landmark' && d0 - this._lastLandmarkD >= LANDMARK_COOLDOWN) {
      landmark = pick(LANDMARKS, this._h(DOMAIN.landmark, index));
      this._lastLandmarkD = d0;
    }

    const len = 250 + this._h(DOMAIN.beat, index, 3) * 550;
    return { index, d0, d1: d0 + len, role, landform, enclosure, vistaSide, routeClass: chapter.routeClass, landmark, biome: chapter.biome };
  }

  beatAt(d) {
    this._ensureBeats(d);
    for (let i = this._beats.length - 1; i >= 0; i--) if (d >= this._beats[i].d0) return this._beats[i];
    return this._beats[0];
  }

  /** Beats overlapping [d0, d1] — for the debug timeline and range queries. */
  beatsInRange(d0, d1) {
    this._ensureBeats(d1);
    return this._beats.filter((b) => b.d1 >= d0 && b.d0 <= d1);
  }

  /** Enclosure 0..1 at a road distance — scales scatter density (open vistas vs forest). */
  enclosureAt(d) { return this.beatAt(d).enclosure; }
}
