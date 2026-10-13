"""Card art sprites for the card atlas: python scripts/fetch_card_art.py [SET ...]

For every card the atlas can show (hand_value_v2 card list + the synergy axes' cards), downloads Scryfall's art crop
(the set's own printing when it has one, else the latest), scales it to TW x TH and packs one JPEG sprite sheet per
set. Artifact pages can't load images from other sites, so the sheets ship with the Artifact; the standalone site
links Scryfall's CDN instead (the art_crop URL is kept per card). Downloads are cached in ~/.cache/mtg-card-art.
Writes reports/art/{SET}.jpg and reports/art/index.json ({set: {card: [cell, artist, art_crop url]}}, plus the grid
layout), which build_card_atlas.py embeds.
"""
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decks import ROOT  # noqa: E402

SETS = ("BLB", "DFT", "DSK", "EOE", "FDN", "FIN", "LCI", "MKM", "MSH", "OTJ", "SOS", "TDM")
TW, TH, COLS = 200, 146, 16
CACHE = Path.home() / ".cache" / "mtg-card-art"
HEADERS = {"User-Agent": "mtg-deck-strength/0.1 (card atlas)", "Accept": "*/*"}
NAME_FIXES = {"Bespoke B?": "Bespoke Bō"}


def fetch(url, **params):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        return r.read()


def get(url, **params):
    for wait in (0.25, 5, 30, 90):        # Scryfall asks for <= 10 requests a second; back off on 429 / 5xx
        time.sleep(wait)
        try:
            return json.loads(fetch(url, **params))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except urllib.error.URLError:
            pass
    raise RuntimeError(f"Scryfall keeps failing: {url} {params}")


def art_uri(card):
    uris = card.get("image_uris") or (card.get("card_faces") or [{}])[0].get("image_uris") or {}
    return uris.get("art_crop"), card.get("artist") or (card.get("card_faces") or [{}])[0].get("artist", "")


def names_for(set_code):
    base = ROOT / "results" / f"{set_code}.PremierDraft"
    names = set(pl.read_csv(base / "hand_value_v2" / "card_values.csv").filter(~pl.col("is_land"))["card"])
    ax = base / "deck_synergy_v8" / "hand_axes.csv"
    if ax.exists():
        names |= set(pl.read_csv(ax)["card"])
    return sorted(names)


def one_set(set_code, index):
    names = names_for(set_code)
    found, url = {}, "https://api.scryfall.com/cards/search"
    page = get(url, q=f"e:{set_code.lower()}", unique="cards")
    while page and page.get("data"):
        for c in page["data"]:
            found.setdefault(c["name"], c)
            found.setdefault(c["name"].split(" // ")[0], c)
        page = get(page["next_page"]) if page.get("has_more") else None
    cache = CACHE / set_code
    cache.mkdir(parents=True, exist_ok=True)
    thumbs, meta, missing = [], {}, []
    for n in names:
        f = cache / (re.sub(r"[^\w-]+", "_", n) + ".jpg")
        info = cache / (f.stem + ".json")
        known = json.loads(info.read_text()) if info.exists() else {}
        if not f.exists() or "uri" not in known:
            c = found.get(n) or found.get(NAME_FIXES.get(n, n)) or get("https://api.scryfall.com/cards/named",
                                                                         exact=NAME_FIXES.get(n, n)) \
                or get("https://api.scryfall.com/cards/named", fuzzy=n.split(" // ")[0])
            uri, artist = art_uri(c) if c and c.get("object") == "card" else (None, known.get("artist", ""))
            if not uri:
                missing.append(n)
                if not f.exists():
                    continue
            if not f.exists():
                time.sleep(0.05)
                f.write_bytes(fetch(uri))
            if uri:
                info.write_text(json.dumps({"artist": artist, "uri": uri}))
        im = Image.open(f).convert("RGB")
        s = max(TW / im.width, TH / im.height)          # cover, centre crop
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        l, t = (im.width - TW) // 2, (im.height - TH) // 2
        thumbs.append(im.crop((l, t, l + TW, t + TH)))
        known = json.loads(info.read_text()) if info.exists() else {}
        meta[n] = [len(thumbs) - 1, known.get("artist", ""), known.get("uri")]
    rows = -(-len(thumbs) // COLS)
    sheet = Image.new("RGB", (COLS * TW, rows * TH))
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % COLS) * TW, (k // COLS) * TH))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=72, optimize=True, progressive=True)
    d = ROOT / "reports" / "art"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{set_code}.jpg").write_bytes(buf.getvalue())
    index[set_code] = meta
    print(f"{set_code}: {len(thumbs)}/{len(names)} cards, sheet {buf.tell() / 1e6:.1f} MB"
          + (f", missing {missing}" if missing else ""), flush=True)


def main(sets):
    path = ROOT / "reports" / "art" / "index.json"
    index = json.loads(path.read_text())["sets"] if path.exists() else {}
    for s in sets:
        one_set(s, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"tile": [TW, TH], "cols": COLS, "sets": index}, separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv[1:] or SETS)
