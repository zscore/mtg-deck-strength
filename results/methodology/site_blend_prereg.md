# Site blend: decision rule (written 2026-10-13, before any vault-set site_blend result was read)

Question: should the draft sim site replace its deck model (the site FM, scored by deck.js) with a blend of it and a
LightGBM?

Primary comparison: `blend* counts + site FM - site FM` (counts-only LightGBM: the cheapest to ship, no structure /
deck-feature / affinity code needed in the LightGBM half). Primary key: `experienced dl`.

1. Dev (12 sets): the repo's decision rule (eval_protocol.decide: RE mean > 0 at p < 0.05, >= 8/12 sets better, no
   set significantly worse, mean >= 0.02 x 1e-3 nats/game).
2. Vault (6 sets, ECL TLA WOE MOM HOB TMT): RE mean > 0 with one-sided HKSJ p < 0.05. These sets were used once by
   the research vault (for the champion, not the site FM), and never for any site choice.
3. Secondary, reported but not deciding: the same for `blend* r4 + site FM`; `blend* r4 + site FM - blend* counts +
   site FM` (is the r4 LightGBM worth its extra browser code?); the fixed alpha 0.7 arm; the match 1 / pre keys.
4. A model that passes 1-2 still has to pass the builder study (site_blend_builds): it must not make builds that are
   visibly worse (mana, colours), and its build-time cost must stay under ~1 s per build.
