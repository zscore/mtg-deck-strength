"""Card text embeddings in DraftEncoder's format (arXiv 2607.04782, Fig. 2), per set.

    python scripts/card_text_embeddings_paper.py MOM WOE ...   (needs sentence-transformers>=5)

Each card becomes one tagged string, e.g.
    [NAME] Quantum Riddler [MC] {3}{U}{U} [CMC] 5 [COLOR] U [TYPE] Creature [SUBTYPE] Sphinx
    [RARITY] mythic [KEYWORDS] Flying, Warp [EFFECTS] Flying <NL> When this creature enters, ... [P/T] 4/6
with [FACE_<i>] before each face of a multi-face card, and is embedded with EmbeddingGemma (768-d), as in
the paper (with its task prompt). Unlike card_text_embeddings.py the name is kept, as the paper does.
Writes data/features/{SET}.PremierDraft/text_emb_gemma.npy, [C, 768] in the draft file's card-column order.
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mtgdata.cardinfo import _scryfall  # noqa: E402
from mtgdata.data import FEATURES, PARQUET, card_metadata  # noqa: E402
from decks import ROOT  # noqa: E402

# google/embeddinggemma-300m is gated on the Hub; the mirrors carry the same weights
MODELS = ("google/embeddinggemma-300m", "unsloth/embeddinggemma-300m")
PROMPT = "task: Magic the Gathering card selection in a draft | card: "  # from the paper's code (nlp.py)


def face_string(f: dict, card: dict, rarity: str, with_name: bool) -> str:
    parts = []
    if with_name:
        parts.append(f"[NAME] {f.get('name', card['name'])}")
    if f.get("mana_cost"):
        parts.append(f"[MC] {f['mana_cost']}")
    if with_name:  # card-level fields once, on the first face
        parts.append(f"[CMC] {card.get('cmc', 0):g}")
    colors = f.get("colors", card.get("colors")) or []
    if colors:
        parts.append(f"[COLOR] {''.join(colors)}")
    main, _, sub = (f.get("type_line") or card.get("type_line", "")).partition(" — ")
    parts.append(f"[TYPE] {main.strip()}")
    if sub:
        parts.append(f"[SUBTYPE] {sub.strip()}")
    if with_name:
        parts.append(f"[RARITY] {rarity}")
        if card.get("keywords"):
            parts.append(f"[KEYWORDS] {', '.join(card['keywords'])}")
    if f.get("oracle_text"):
        parts.append(f"[EFFECTS] {f['oracle_text'].replace(chr(10), ' <NL> ')}")
    if "power" in f:
        parts.append(f"[P/T] {f['power']}/{f['toughness']}")
    if "loyalty" in f:
        parts.append(f"[LOYALTY] {f['loyalty']}")
    return " ".join(parts)


def card_string(card: dict, rarity: str) -> str:
    faces = card.get("card_faces")
    if not faces:
        return face_string(card, card, rarity, True)
    head = f"[NAME] {card['name']} [CMC] {card.get('cmc', 0):g} [RARITY] {rarity}"
    if card.get("keywords"):
        head += f" [KEYWORDS] {', '.join(card['keywords'])}"
    return head + "".join(f" [FACE_{i + 1}] " + face_string(f, card, rarity, False) for i, f in enumerate(faces))


def load_model():
    from sentence_transformers import SentenceTransformer
    for name in MODELS:
        try:
            return name, SentenceTransformer(name, device="cpu")
        except Exception as e:  # gated or missing
            print(f"  {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
    raise SystemExit("no EmbeddingGemma checkpoint available")


def main(*sets):
    name, model = load_model()
    print("model", name, flush=True)
    for s in sets:
        cols = pl.scan_parquet(PARQUET / f"{s}.PremierDraft.parquet").collect_schema().names()
        names = [c[len("pack_card_"):] for c in cols if c.startswith("pack_card_")]
        sc = _scryfall(names, s)
        rarity = dict(zip(names, card_metadata(names, s)["rarity"].to_list()))
        texts = [card_string(sc[n], rarity[n]) if n in sc else f"[NAME] {n} [RARITY] {rarity[n]}" for n in names]
        emb = model.encode([PROMPT + t for t in texts], batch_size=16, normalize_embeddings=True,
                           show_progress_bar=False)
        d = FEATURES / f"{s}.PremierDraft"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "text_emb_gemma.npy", emb.astype(np.float32))
        (d / "card_strings.txt").write_text("\n".join(texts))
        print(s, emb.shape, "missing", sum(n not in sc for n in names), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
