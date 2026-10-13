"""The four synergy axes of round 8's opening-hand pair model: python scripts/synergy_axes.py [SET ...]

deck_synergy_v8's slope + pairs k=4 model scores a pair as Q_ij = <v_i, v_j>, v_i in R^4, but saves only Q (the pairs
in >= 2,000 training decks, hand_pairs.csv). Q is exactly rank 4 and PSD, so V is recovered from those entries by
least squares, up to a rotation (the model itself is rotation-invariant). The rotation here is the principal one:
axis 1 carries the most synergy (largest eigenvalue of V'V), and each axis's sign puts its largest loading positive.
Cards pointing the same way synergise, opposite ways anti-synergise, at right angles don't interact.
Writes results/{SET}.PremierDraft/deck_synergy_v8/hand_axes.csv (card, a1..a4, norm) and hand_axes.json (fit stats).
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decks import ROOT  # noqa: E402

SETS = ("BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "MSH", "OTJ", "SOS", "TDM")
K = 4


def recover(i, j, q, n, seed):
    rng = np.random.default_rng(seed)

    def f(x):
        V = x.reshape(n, K)
        r = (V[i] * V[j]).sum(1) - q
        g = np.zeros_like(V)
        np.add.at(g, i, r[:, None] * V[j])
        np.add.at(g, j, r[:, None] * V[i])
        return 0.5 * (r @ r), g.ravel()
    res = minimize(f, 0.1 * rng.standard_normal(n * K), jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "gtol": 1e-12, "ftol": 1e-16})
    return res.x.reshape(n, K), res.fun


def one_set(set_code):
    out = ROOT / "results" / f"{set_code}.PremierDraft" / "deck_synergy_v8"
    p = pl.read_csv(out / "hand_pairs.csv")
    names = sorted(set(p["card a"]) | set(p["card b"]))
    ix = {c: t for t, c in enumerate(names)}
    i = np.array([ix[c] for c in p["card a"]])
    j = np.array([ix[c] for c in p["card b"]])
    q = p["Q"].to_numpy().astype(np.float64)
    V, loss = min((recover(i, j, q, len(names), s) for s in range(3)), key=lambda t: t[1])
    fit = (V[i] * V[j]).sum(1)
    ev, R = np.linalg.eigh(V.T @ V)
    order = np.argsort(ev)[::-1]
    V, ev = V @ R[:, order], ev[order]
    V *= np.sign(V[np.abs(V).argmax(0), range(K)])
    stats = {"set": set_code, "cards": len(names), "pairs": len(q), "fit r": float(np.corrcoef(fit, q)[0, 1]),
             "rmse": float(np.sqrt(np.mean((fit - q) ** 2))), "q sd": float(q.std()),
             "axis share": (ev / ev.sum()).round(4).tolist()}
    pl.DataFrame({"card": names, **{f"a{k + 1}": V[:, k] for k in range(K)},
                  "norm": np.linalg.norm(V, axis=1)}).sort("norm", descending=True).write_csv(out / "hand_axes.csv")
    (out / "hand_axes.json").write_text(json.dumps(stats, indent=1))
    print(f"{set_code}: {stats['cards']} cards, fit r {stats['fit r']:.5f}, rmse {stats['rmse']:.2e} "
          f"(Q sd {stats['q sd']:.3f}), axis share {stats['axis share']}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or SETS:
        one_set(s)
