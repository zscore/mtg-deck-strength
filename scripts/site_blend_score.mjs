// Score decks with the site's deck model (deck.js DeckModel), exactly as the browser does.
//   node score_site_fm.mjs SET EVENT decks.bin out.bin [model.json]
// model.json: another file in data/deck/{SET}.json's format (default: the site's own); EVENT is any key of it.
// decks.bin: int8 [N, C] counts in data/deck/{SET}.json `names` order (C = names.length).
// out.bin: float64 [N, 2]: clamped logit (what the builder uses), raw logit (unclamped).
import { readFileSync, writeFileSync } from 'node:fs';
import { DeckModel } from '/Users/zblanton/Documents/repos/mtg-draft-sim-site/deck.js';

const [set, event, inp, out, file] = process.argv.slice(2);
const json = JSON.parse(readFileSync(file ?? `/Users/zblanton/Documents/repos/mtg-draft-sim-site/data/deck/${set}.json`));
// a stand-in set whose card ids are the model's own name indices, so every card has its trained row
const fake = { cards: Object.fromEntries(json.names.map((n, i) => [i, { name: n, type: '', colors: '' }])) };
const model = new DeckModel(json, event, fake);
const C = json.names.length, buf = readFileSync(inp), N = buf.length / C;
if (!Number.isInteger(N)) throw new Error(`bad size ${buf.length} for C=${C}`);
const res = new Float64Array(2 * N), x = new Int8Array(buf.buffer, buf.byteOffset, buf.length);
const t0 = performance.now();
for (let r = 0; r < N; r++) {
  const m = new Map();
  for (let j = 0; j < C; j++) { const n = x[r * C + j]; if (n) m.set(j, n); }
  const g = model.aggregate(m);
  res[2 * r] = model.logit(g);
  res[2 * r + 1] = model.rawLogit(g);
}
writeFileSync(out, Buffer.from(res.buffer));
console.log(`${set} ${event}: ${N} decks scored in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
