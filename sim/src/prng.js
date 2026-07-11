// Alea PRNG — deterministic, seedable, fast. Same seed always yields the same
// stream, which is the property the whole oracle harness depends on.

export function createPrng(seed) {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let c = 1;

  const mash = makeMash();
  s0 = mash(" ");
  s1 = mash(" ");
  s2 = mash(" ");

  const seedStr = String(seed);
  s0 -= mash(seedStr);
  if (s0 < 0) s0 += 1;
  s1 -= mash(seedStr);
  if (s1 < 0) s1 += 1;
  s2 -= mash(seedStr);
  if (s2 < 0) s2 += 1;

  function next() {
    const t = 2091639 * s0 + c * 2.3283064365386963e-10; // 2^-32
    s0 = s1;
    s1 = s2;
    return (s2 = t - (c = t | 0));
  }

  return {
    next,
    // Uniform in [min, max).
    range: (min, max) => min + next() * (max - min),
    // Signed uniform in [-mag, mag).
    signed: (mag = 1) => (next() * 2 - 1) * mag,
  };
}

function makeMash() {
  let n = 0xefc8249d;
  return function mash(data) {
    data = String(data);
    for (let i = 0; i < data.length; i++) {
      n += data.charCodeAt(i);
      let h = 0.02519603282416938 * n;
      n = h >>> 0;
      h -= n;
      h *= n;
      n = h >>> 0;
      h -= n;
      n += h * 0x100000000; // 2^32
    }
    return (n >>> 0) * 2.3283064365386963e-10; // 2^-32
  };
}
