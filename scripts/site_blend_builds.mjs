// Builder study for site_blend: the site's deck builder (deck.js buildDeck) with its FM and with blend* (the FM
// plus the counts-only LightGBM of site_blend.py), on real test-draft pools.
//   node scripts/site_blend_builds.mjs SET [--pools N] [--alpha A]
// Reads results/{SET}.PremierDraft/site_blend/{builder.json, lgbm_counts_seed0.txt}; writes builds.json there.
import { readFileSync, writeFileSync } from 'node:fs';
import { gzipSync } from 'node:zlib';

const SITE = '/Users/zblanton/Documents/repos/mtg-draft-sim-site';
const { DeckModel, buildDeck, cardValues, deckPerColors, isLand } = await import(`${SITE}/deck.js`);
const REPO = new URL('..', import.meta.url).pathname;

// ---------------------------------------------------------------- LightGBM text model -> flat arrays

export function parseLgbm(text) {
  const trees = [];
  for (const block of text.split(/\nTree=\d+\n/).slice(1)) {
    const f = {};
    for (const line of block.split('\n')) {
      const i = line.indexOf('=');
      if (i > 0) f[line.slice(0, i)] = line.slice(i + 1);
      if (line.startsWith('end of trees')) break;
    }
    const nums = (k) => (f[k] ? f[k].trim().split(' ').map(Number) : []);
    const t = { feat: nums('split_feature'), thr: nums('threshold'), left: nums('left_child'), right: nums('right_child'),
      leaf: nums('leaf_value'), dt: nums('decision_type') };
    for (const d of t.dt) if (d & 1) throw new Error('categorical split'); // numeric "<=" only
    trees.push(t);
  }
  return trees;
}

// one flat layout: per node feature (u16), integer threshold (counts: x <= thr), children (>= 0 node, < 0 ~leaf)
export function compact(trees) {
  let nodes = 0, leaves = 0;
  for (const t of trees) { nodes += t.feat.length; leaves += t.leaf.length; }
  const feat = new Uint16Array(nodes), thr = new Uint8Array(nodes), kids = new Int32Array(2 * nodes);
  const leaf = new Float32Array(leaves), root = new Int32Array(trees.length);
  let n0 = 0, l0 = 0;
  trees.forEach((t, k) => {
    root[k] = t.feat.length ? n0 : ~l0;
    for (let i = 0; i < t.feat.length; i++) {
      feat[n0 + i] = t.feat[i];
      thr[n0 + i] = Math.floor(t.thr[i]); // counts are integers: x <= 1.0000000180025095e-35 is x <= 0
      for (const [j, c] of [[0, t.left[i]], [1, t.right[i]]]) kids[2 * (n0 + i) + j] = c >= 0 ? n0 + c : ~(l0 + ~c);
    }
    leaf.set(t.leaf, l0);
    n0 += t.feat.length; l0 += t.leaf.length;
  });
  return { feat, thr, kids, leaf, root };
}

export function lgbmScore(m, x, off = 0) {
  let z = 0;
  for (let k = 0; k < m.root.length; k++) {
    let n = m.root[k];
    while (n >= 0) n = x[off + m.feat[n]] <= m.thr[n] ? m.kids[2 * n] : m.kids[2 * n + 1];
    z += m.leaf[~n];
  }
  return z;
}

function treeOut(m, k, x, off) {
  let n = m.root[k];
  while (n >= 0) n = x[off + m.feat[n]] <= m.thr[n] ? m.kids[2 * n] : m.kids[2 * n + 1];
  return m.leaf[~n];
}

// Incremental LightGBM: keeps every tree's output for a base deck and, for a deck that differs from it in a few
// card counts, re-walks only the trees that split on those cards. A deck that differs in more than `maxChanged`
// counts is walked in full and becomes the new base (the builder scores many neighbours of one deck in a row).
export class IncLgbm {
  constructor(m, nFeat, maxChanged = 6) {
    const lists = Array.from({ length: nFeat }, () => new Set());
    const nTrees = m.root.length;
    for (let k = 0; k < nTrees; k++) {
      const stack = m.root[k] >= 0 ? [m.root[k]] : [];
      while (stack.length) {
        const n = stack.pop();
        lists[m.feat[n]].add(k);
        for (const c of [m.kids[2 * n], m.kids[2 * n + 1]]) if (c >= 0) stack.push(c);
      }
    }
    this.treesOf = lists.map((s) => Int32Array.from(s));
    Object.assign(this, { m, nFeat, maxChanged, nTrees });
    this.base = new Float64Array(nFeat).fill(-1);
    this.out = new Float64Array(nTrees);
    this.sum = 0;
    this.stamp = new Int32Array(nTrees);
    this.tick = 0;
    this.changed = new Int32Array(nFeat);
    this.stats = { full: 0, inc: 0, trees: 0 };
  }

  score(x, off) {
    let nc = 0;
    for (let f = 0; f < this.nFeat; f++) {
      if (x[off + f] !== this.base[f]) { if (nc === this.maxChanged) { nc = -1; break; } this.changed[nc++] = f; }
    }
    const { m } = this;
    if (nc < 0) { // full walk, new base
      let s = 0;
      for (let k = 0; k < this.nTrees; k++) { const v = treeOut(m, k, x, off); this.out[k] = v; s += v; }
      for (let f = 0; f < this.nFeat; f++) this.base[f] = x[off + f];
      this.sum = s;
      this.stats.full++;
      return s;
    }
    let s = this.sum;
    const t = ++this.tick;
    for (let i = 0; i < nc; i++) {
      for (const k of this.treesOf[this.changed[i]]) {
        if (this.stamp[k] === t) continue;
        this.stamp[k] = t;
        s += treeOut(m, k, x, off) - this.out[k];
        this.stats.trees++;
      }
    }
    this.stats.inc++;
    return s;
  }
}

// ---------------------------------------------------------------- blend* as a DeckModel

// The FM's aggregate plus the deck's card counts in LightGBM feature order (slots Dfm..), so deck.js's incremental
// rescoring carries the counts along; logit() is the blend, put back on the FM's logit scale (mu_F + sd_F * z) so
// the builder's penalties and winRate keep their meaning.
class BlendModel extends DeckModel {
  constructor(json, set, lg, featOf, nFeat, cfg) {
    super(json, 'draft', set);
    this.Dfm = this.D;
    this.D = this.Dfm + nFeat;
    Object.assign(this, { lg, featOf, cfg });
  }

  add(g, id, n, have = null) {
    const r = this.rows.get(id);
    for (let d = 0; d < this.Dfm; d++) g.s[d] += n * r.row[d];
    const h = have ?? 0, dsq = (h + n) ** 2 - h ** 2;
    g.vv += dsq * r.vv;
    g.ae += dsq * r.ae;
    const f = this.featOf.get(id);
    if (f !== undefined) g.s[this.Dfm + f] += n;
  }

  parts(g) {
    return { fm: super.logit(g), lg: this.inc ? this.inc.score(g.s, this.Dfm) : lgbmScore(this.lg, g.s, this.Dfm) };
  }

  logit(g) {
    const { fm, lg } = this.parts(g), c = this.cfg;
    const z = c.alpha * (fm - c.muF) / c.sdF + (1 - c.alpha) * (lg - c.muL) / c.sdL;
    return c.muF + c.sdF * z;
  }
}

function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------- run

const args = process.argv.slice(2);
const code = args[0];
const opt = (k, d) => (args.includes(k) ? Number(args[args.indexOf(k) + 1]) : d);
const dir = `${REPO}results/${code}.PremierDraft/site_blend`;
const B = JSON.parse(readFileSync(`${dir}/builder.json`));
const json = JSON.parse(readFileSync(`${SITE}/data/deck/${code}.json`));
const set = JSON.parse(readFileSync(`${SITE}/data/${code}.json`));
const idOf = new Map(Object.entries(set.cards).map(([id, c]) => [c.name, +id]));
const featOf = new Map();
B.names.forEach((n, i) => { if (idOf.has(n)) featOf.set(idOf.get(n), i); });
const unmappedFeat = B.names.filter((n) => !idOf.has(n));

const text = readFileSync(`${dir}/lgbm_counts_seed0.txt`, 'utf8');
const trees = parseLgbm(text), lg = compact(trees);
const alpha = opt('--alpha', B['alpha counts (FM weight)']);
const cfg = { alpha, muF: B['site FM'][0], sdF: B['site FM'][1], muL: B['LGBM counts'][0], sdL: B['LGBM counts'][1] };
const fm = new DeckModel(json, 'draft', set);
const bl = new BlendModel(json, set, lg, featOf, B.names.length, cfg);

// sizes of the shipped form
const f16 = new Uint16Array(lg.leaf.length); // float16 by truncation-rounding, enough for a size estimate
const bin = Buffer.concat([lg.feat, lg.thr, lg.kids, lg.leaf].map((a) => Buffer.from(a.buffer)));
const kids16 = Int16Array.from(lg.kids);
const bin16 = Buffer.concat([lg.feat, lg.thr, kids16, f16, lg.root].map((a) => Buffer.from(a.buffer)));
const sizes = { trees: trees.length, nodes: lg.feat.length, leaves: lg.leaf.length,
  'text kB': text.length / 1024, 'f32 int32-kids kB': bin.length / 1024, 'f32 gz kB': gzipSync(bin, { level: 9 }).length / 1024,
  'f16 int16-kids kB': bin16.length / 1024, 'site deck json kB': readFileSync(`${SITE}/data/deck/${code}.json`).length / 1024,
  'site deck json gz kB': gzipSync(readFileSync(`${SITE}/data/deck/${code}.json`), { level: 9 }).length / 1024 };

const toCounts = (obj) => {
  const m = new Map();
  for (const [n, k] of Object.entries(obj)) if (idOf.has(n)) m.set(idOf.get(n), (m.get(idOf.get(n)) || 0) + k);
  return m;
};
const named = (counts) => Object.fromEntries([...counts].map(([id, n]) => [set.cards[id].name, n]));
const evalAll = (counts) => {
  const g = bl.aggregate(counts), p = bl.parts(g);
  return { fm: p.fm, lg: p.lg, blend: bl.logit(g) };
};

if (args.includes('--timing')) {
  // FM vs full blend vs incremental blend on the same pools; the incremental build must pick the same deck
  const K = opt('--timing', 40);
  const bi = new BlendModel(json, set, lg, featOf, B.names.length, cfg);
  bi.inc = new IncLgbm(lg, B.names.length);
  const t = [0, 0, 0];
  let same = 0, n = 0, maxErr = 0;
  for (const p of B.pools.slice(0, K)) {
    const pool = [];
    for (const [nm, k] of Object.entries(p.pool)) if (idOf.has(nm)) for (let i = 0; i < k; i++) pool.push(idOf.get(nm));
    const seed = [...p.draft_id].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7);
    const res = [];
    for (const [j, mdl] of [fm, bl, bi].entries()) {
      const t0 = performance.now();
      res.push(buildDeck(mdl, pool, { rng: mulberry32(seed) }));
      t[j] += performance.now() - t0;
    }
    same += JSON.stringify([...res[1].counts].sort()) === JSON.stringify([...res[2].counts].sort());
    maxErr = Math.max(maxErr, Math.abs(res[1].logit - res[2].logit));
    n++;
  }
  // hybrid: the FM's full search, then one blend search aimed at the FM deck's main colours (any splash)
  if (args.includes('--polish')) {
    const r = { n: 0, tFm: 0, tPol: 0, tBl: 0, gainFull: 0, gainPol: 0, samePolFull: 0, polNull: 0 };
    const sideCols = (counts) => {
      const n = {};
      for (const [id, k] of counts) {
        const c = set.cards[id];
        if (isLand(c)) continue;
        for (const col of new Set((c.colors || '').split(''))) if (col) n[col] = (n[col] || 0) + k;
      }
      return Object.keys(n).filter((col) => n[col] >= 4).sort((a, b) => 'WUBRG'.indexOf(a) - 'WUBRG'.indexOf(b)).join('');
    };
    for (const p of B.pools.slice(0, K)) {
      const pool = [];
      for (const [nm, k] of Object.entries(p.pool)) if (idOf.has(nm)) for (let i = 0; i < k; i++) pool.push(idOf.get(nm));
      const seed = [...p.draft_id].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7);
      let t0 = performance.now();
      const a = buildDeck(fm, pool, { rng: mulberry32(seed) });
      r.tFm += performance.now() - t0;
      t0 = performance.now();
      const main = sideCols(a.counts);
      let pol = main.length >= 1 && main.length <= 3
        ? buildDeck(bl, pool, { kicks: 2, rng: mulberry32(seed), want: { main, splash: '*' } }) : null;
      r.tPol += performance.now() - t0;
      t0 = performance.now();
      const b = buildDeck(bl, pool, { rng: mulberry32(seed) });
      r.tBl += performance.now() - t0;
      const sc = (x) => bl.logit(bl.aggregate(x.counts));
      if (!pol || sc(pol) < sc(a)) { r.polNull += !pol; pol = a; } // never worse than the FM deck under the blend
      const wr = (z) => 100 / (1 + Math.exp(-(fm.base + z)));
      r.gainFull += wr(sc(b)) - wr(sc(a));
      r.gainPol += wr(sc(pol)) - wr(sc(a));
      r.samePolFull += JSON.stringify([...pol.counts].sort()) === JSON.stringify([...b.counts].sort());
      r.n++;
    }
    const n = r.n;
    console.log(JSON.stringify({ set: code, pools: n, 'fm ms': r.tFm / n, 'polish ms (extra)': r.tPol / n,
      'full blend ms': r.tBl / n, 'blend gain over FM deck, full (pp)': r.gainFull / n,
      'blend gain over FM deck, polish (pp)': r.gainPol / n, 'polish = full blend deck': r.samePolFull / n,
      'polish found nothing': r.polNull / n }));
    process.exit(0);
  }
  // sealed-sized pools (six random packs from the site's collation), FM vs blend
  if (args.includes('--sealed')) {
    const { openPack } = await import(`${SITE}/packs.js`);
    const ts = [0, 0, 0, 0];
    const NS = 10;
    for (let r = 0; r < NS; r++) {
      const pool = Array.from({ length: 6 }, () => openPack(set)).flat();
      for (const [j, mdl] of [fm, bl].entries()) {
        let t0 = performance.now();
        buildDeck(mdl, pool, { rng: mulberry32(r + 1) });
        ts[j] += performance.now() - t0;
        if (r < 3) {
          t0 = performance.now();
          for (const _ of deckPerColors(mdl, pool, (key) => mulberry32(key.length * 7919))) { /* drain */ }
          ts[2 + j] += performance.now() - t0;
        }
      }
    }
    console.log(JSON.stringify({ set: code, 'sealed build fm ms': ts[0] / NS, 'sealed build blend ms': ts[1] / NS,
      'sealed colours fm ms': ts[2] / 3, 'sealed colours blend ms': ts[3] / 3 }));
    process.exit(0);
  }
  // the "best deck by colours" panel: 30 targeted searches per pool
  const tc = [0, 0];
  const nc = Math.min(8, K);
  for (const p of B.pools.slice(0, nc)) {
    const pool = [];
    for (const [nm, k] of Object.entries(p.pool)) if (idOf.has(nm)) for (let i = 0; i < k; i++) pool.push(idOf.get(nm));
    for (const [j, mdl] of [fm, bl].entries()) {
      const t0 = performance.now();
      for (const _ of deckPerColors(mdl, pool, (key) => mulberry32(key.length * 7919))) { /* drain */ }
      tc[j] += performance.now() - t0;
    }
  }
  const s = bi.inc.stats;
  console.log(JSON.stringify({ 'by colours fm ms': tc[0] / nc, 'by colours blend ms': tc[1] / nc, set: code, trees: trees.length, pools: n, 'fm ms': t[0] / n, 'blend full ms': t[1] / n,
    'blend incremental ms': t[2] / n, 'incremental = full (same deck)': same / n, 'max |logit diff|': maxErr,
    'full walks': s.full, 'incremental walks': s.inc, 'trees per incremental walk': s.trees / Math.max(s.inc, 1) }));
  process.exit(0);
}

const nPools = opt('--pools', B.pools.length);
const out = { set: code, cfg, sizes, unmappedFeat, base: fm.base, pools: [] };
let tFm = 0, tBl = 0, tVal = [0, 0];
// timing of the scorer alone
{
  const decks = B.pools.slice(0, 200).map((p) => toCounts(p.deck));
  const gs = decks.map((x) => bl.aggregate(x));
  let t = performance.now();
  for (let r = 0; r < 20; r++) for (const g of gs) fm.logit(g);
  const tf = (performance.now() - t) / (20 * gs.length);
  t = performance.now();
  for (let r = 0; r < 20; r++) for (const g of gs) lgbmScore(lg, g.s, bl.Dfm);
  out.timing = { 'fm logit us': 1000 * tf, 'lgbm us': 1000 * (performance.now() - t) / (20 * gs.length) };
  // registered decks: JS LightGBM vs Python's (checked in site_blend_analyze.py)
  out.registered = decks.map((x) => evalAll(x));
  out.registeredIds = B.pools.slice(0, 200).map((p) => p.draft_id);
}
for (const p of B.pools.slice(0, nPools)) {
  const pool = [];
  for (const [n, k] of Object.entries(p.pool)) if (idOf.has(n)) for (let i = 0; i < k; i++) pool.push(idOf.get(n));
  const seed = [...p.draft_id].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7);
  let t = performance.now();
  const a = buildDeck(fm, pool, { rng: mulberry32(seed) });
  tFm += performance.now() - t;
  t = performance.now();
  const b = buildDeck(bl, pool, { rng: mulberry32(seed) });
  tBl += performance.now() - t;
  let vt = performance.now();
  // search-noise baseline: the FM again with other random kicks
  const a2 = buildDeck(fm, pool, { rng: mulberry32(seed ^ 0x5bd1e995) });
  const valsF = out.pools.length < 30 ? cardValues(fm, a.counts, pool) : null;
  tVal[0] += out.pools.length < 30 ? performance.now() - vt : 0;
  vt = performance.now();
  const valsB = out.pools.length < 30 ? cardValues(bl, b.counts, pool) : null;
  tVal[1] += out.pools.length < 30 ? performance.now() - vt : 0;
  const reg = toCounts(p.deck);
  out.pools.push({
    draft_id: p.draft_id, skill: p.skill, wins: p.wins, losses: p.losses,
    poolMissing: Object.keys(p.pool).filter((n) => !idOf.has(n)),
    fmDeck: named(a.counts), blDeck: named(b.counts), regDeck: named(reg),
    fm2Deck: named(a2.counts), fm2Built: evalAll(a2.counts),
    fmBuilt: evalAll(a.counts), blBuilt: evalAll(b.counts), reg: evalAll(reg),
    nLands: [a, b].map((d) => [...d.counts].reduce((s, [id, n]) => s + (isLand(set.cards[id]) ? n : 0), 0)),
    values: valsF && { fm: Object.fromEntries([...valsF].map(([id, v]) => [set.cards[id].name, v])),
      bl: Object.fromEntries([...valsB].map(([id, v]) => [set.cards[id].name, v])) },
  });
}
const n = out.pools.length;
out.timing = { ...out.timing, 'fm build ms': tFm / n, 'blend build ms': tBl / n,
  'fm values ms': tVal[0] / Math.min(30, n), 'blend values ms': tVal[1] / Math.min(30, n) };
writeFileSync(`${dir}/builds.json`, JSON.stringify(out));
console.log(`${code}: ${n} pools, ${trees.length} trees; build ${out.timing['fm build ms'].toFixed(0)} ms (FM) vs ` +
  `${out.timing['blend build ms'].toFixed(0)} ms (blend); scorer ${out.timing['fm logit us'].toFixed(1)} / ` +
  `${out.timing['lgbm us'].toFixed(1)} us; f16 ${sizes['f16 int16-kids kB'].toFixed(0)} kB; unmapped ${unmappedFeat.length}`);
