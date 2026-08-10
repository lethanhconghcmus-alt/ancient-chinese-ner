"""Build TITLE/ORG/LOC gazetteers as a *feature* source (not weak-label source).

Inputs:
  - data/processed/ner_clean/train.jsonl      gold entities (train split ONLY,
    to avoid leaking dev/test entity identity into a lexicon later used as a
    feature when evaluating on dev/test — see docs/gazetteer_findings.md §11)
  - data/raw/gazetteer/dvsktt_sentences.jsonl  full DVSKTT crawl (unlabeled)
  - data/processed/ner_clean/gazetteer_blocklist.txt  2,038 sentences that
    overlap dev/test verbatim — excluded from LOC mining (leakage guard)

Strategy per docs/gazetteer_findings.md §9:
  TITLE — gold train entities + prefix×suffix cross-product (compounds are
          closed-vocabulary; convention is to keep the FULL compound as one
          unit, matching docs/annotation_guideline.md §2)
  ORG   — gold train entities only (1-char ORG needs a context gate that is
          a downstream *feature-usage* concern, not a gazetteer-build one)
  LOC   — mined from the crawl via phienam-capitalization <-> han syllable
          alignment (§4), with the suffix-recovery patch for the 44.9% of
          LOC mentions that lose their generic-noun suffix in phienam
          capitalization (§4 "Cách vá")

Output: data/processed/gazetteer/{title,org,loc}.jsonl + report.json
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_SENT = ROOT / "data/raw/gazetteer/dvsktt_sentences.jsonl"
TRAIN = ROOT / "data/processed/ner_clean/train.jsonl"
DEV = ROOT / "data/processed/ner_clean/dev.jsonl"
BLOCKLIST = ROOT / "data/processed/ner_clean/gazetteer_blocklist.txt"
OUT_DIR = ROOT / "data/processed/gazetteer"

ENTITY_RE = re.compile(r"\{([^{}|]+)\|([A-Z]+)\}")

# §4 "Cách vá": Viet generic-noun suffix -> paired Han suffix char, same order.
SUFFIX_PAIRS = list(zip(
    "châu huyện phủ xã sơn giang thành quận trấn lộ động sách trang "
    "hải khẩu quan tân độ kiều cung điện môn".split(),
    "州縣府社山江城郡鎮路洞柵莊海口關津渡橋宫殿門",
))
SUFFIX_WORD_TO_HAN = {w: h for w, h in SUFFIX_PAIRS}

# Common Vietnamese clan-surname chars (curated, general knowledge — not
# mined from any split, so no leakage risk). Capitalization can't tell PER
# from LOC (docs/gazetteer_findings.md §5-6), so a mined span opening with
# one of these is almost always a person's name and must NOT go into LOC.
SURNAME_CHARS = set(
    "阮鄭黎陳莫吳范武杜李楊裴陶鄧潘丁黄郭張梁何胡蘇謝韓馬高林秦金"
)


def strip_punct(tok: str) -> str:
    return tok.strip(".,;:!?\"'()[]{}·。，、；：！？“”‘’")


def is_capitalized(tok: str) -> bool:
    tok = strip_punct(tok)
    if not tok:
        return False
    return tok[0].isupper()


def load_gold_entities(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for surface, etype in ENTITY_RE.findall(rec["output"]):
                out.append((surface, etype))
    return out


def build_title_org(train_entities):
    title_counter = Counter()
    org_counter = Counter()
    for surface, etype in train_entities:
        if etype == "TITLE":
            title_counter[surface] += 1
        elif etype == "ORG":
            org_counter[surface] += 1

    # TITLE prefix x suffix cross-product over multi-char gold titles.
    multi = [s for s in title_counter if len(s) >= 2]
    prefix_count = Counter(s[0] for s in multi)
    suffix_count = Counter(s[-1] for s in multi)
    # Require the char to open/close >=5 distinct gold TITLE entities: keeps
    # true structural morphemes (太/少/大/上..., 公/王/侯/伯...) instead of
    # every char that happens to recur by chance.
    MIN_FREQ = 5
    prefixes = {c for c, n in prefix_count.items() if n >= MIN_FREQ}
    suffixes = {c for c, n in suffix_count.items() if n >= MIN_FREQ}

    title_rows = [
        {"surface": s, "type": "TITLE", "freq": n, "source": "gold_train", "needs_review": False}
        for s, n in title_counter.items()
    ]
    seen = set(title_counter)
    for p in sorted(prefixes):
        for s in sorted(suffixes):
            cand = p + s
            if cand in seen:
                continue
            seen.add(cand)
            title_rows.append({
                "surface": cand, "type": "TITLE", "freq": 0,
                "source": "crossproduct", "needs_review": True,
            })

    org_rows = [
        {"surface": s, "type": "ORG", "freq": n, "source": "gold_train", "needs_review": False}
        for s, n in org_counter.items()
    ]

    return title_rows, org_rows, sorted(prefixes), sorted(suffixes)


def load_blocklist():
    with BLOCKLIST.open(encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f}


def mine_loc(blocklist, exclude_surfaces):
    loc_counter = Counter()
    n_sent = 0
    n_used = 0
    n_aligned = 0
    with RAW_SENT.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_sent += 1
            if row["han"] in blocklist:
                continue
            n_used += 1
            han = row["han_chars"]
            toks = row["phienam"].split()
            if len(toks) != len(han):
                continue
            n_aligned += 1
            cap = [is_capitalized(t) for t in toks]
            i = 0
            while i < len(cap):
                if not cap[i]:
                    i += 1
                    continue
                j = i
                while j < len(cap) and cap[j]:
                    j += 1
                span = "".join(han[i:j])
                # suffix-recovery patch: next token is a known generic-noun
                # suffix word AND the paired Han char matches at han[j].
                if j < len(toks) and j < len(han):
                    nxt = strip_punct(toks[j]).lower()
                    han_char = SUFFIX_WORD_TO_HAN.get(nxt)
                    if han_char is not None and han[j] == han_char:
                        span = span + han_char
                if (
                    len(span) >= 2
                    and span[0] not in SURNAME_CHARS
                    and span not in exclude_surfaces
                    and not span.endswith("宗")  # temple-name suffix -> PER, not LOC
                ):
                    loc_counter[span] += 1
                i = j

    loc_rows = [
        {"surface": s, "type": "LOC", "freq": n, "source": "mined_crawl", "needs_review": n < 2}
        for s, n in loc_counter.items()
    ]
    stats = {
        "sentences_total": n_sent,
        "sentences_blocklisted": n_sent - n_used,
        "sentences_used": n_used,
        "sentences_aligned": n_aligned,
        "loc_unique": len(loc_counter),
        "loc_mentions": sum(loc_counter.values()),
    }
    return loc_rows, stats


def coverage(rows, etype, dev_entities):
    lex = {r["surface"] for r in rows}
    gold = [s for s, t in dev_entities if t == etype]
    if not gold:
        return None
    hit = sum(1 for s in gold if s in lex)
    return {"dev_mentions": len(gold), "hit": hit, "coverage": round(hit / len(gold), 4)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_entities = load_gold_entities(TRAIN)
    dev_entities = load_gold_entities(DEV)
    blocklist = load_blocklist()

    title_rows, org_rows, prefixes, suffixes = build_title_org(train_entities)
    # Cross-scrub against train gold PER/TITLE/ORG surfaces: capitalization
    # can't tell PER from LOC (§5-6), so any mined span that is a *known*
    # PER/TITLE/ORG surface in gold is dropped rather than mislabeled LOC.
    exclude_surfaces = {s for s, t in train_entities if t in ("PER", "TITLE", "ORG")}
    loc_rows, loc_stats = mine_loc(blocklist, exclude_surfaces)

    for name, rows in (("title", title_rows), ("org", org_rows), ("loc", loc_rows)):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "train_entities_total": len(train_entities),
        "title": {
            "gold_train": sum(1 for r in title_rows if r["source"] == "gold_train"),
            "crossproduct": sum(1 for r in title_rows if r["source"] == "crossproduct"),
            "prefixes": prefixes,
            "suffixes": suffixes,
            "dev_coverage": coverage(title_rows, "TITLE", dev_entities),
        },
        "org": {
            "gold_train": len(org_rows),
            "dev_coverage": coverage(org_rows, "ORG", dev_entities),
        },
        "loc": {
            **loc_stats,
            "dev_coverage": coverage(loc_rows, "LOC", dev_entities),
        },
    }
    with (OUT_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
