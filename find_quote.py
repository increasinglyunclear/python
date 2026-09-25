"""Given a snippet of text, find where it was said in a specific recording. 

by Kevin Walker, Aug 2026. Created for https://the-making-of-creativity.com/

Searches the fused/translated transcript (out/master.json) for the best-matching
segment(s), then uses the audio-alignment offsets (work/sessions.json) to convert
the session-timeline time back into a timestamp inside each contributing raw
recording file in recordings/.

Usage:
    python -m analysis.find_quote "some phrase from the transcript"
    python -m analysis.find_quote "some phrase" --top 5
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

from .config import CONFIG


def _hms(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", " ", s.lower()).strip()


def _word_score(query_words: list[str], cand_words: list[str]) -> float:
    """Fuzzy 'does this phrase live inside this (longer) segment' score.

    Combines: longest contiguous run of query words found in the segment
    (word order matters) with plain word recall (order-agnostic), so a
    paraphrased or partially-misremembered quote still scores well.
    """
    if not query_words or not cand_words:
        return 0.0
    sm = difflib.SequenceMatcher(None, cand_words, query_words, autojunk=False)
    run = sm.find_longest_match(0, len(cand_words), 0, len(query_words))
    contiguous = run.size / len(query_words)
    recall = len(set(query_words) & set(cand_words)) / len(query_words)
    return 0.65 * contiguous + 0.35 * recall


def _score(query: str, seg: dict) -> float:
    """Best of: substring hit, whole-string fuzzy ratio, or partial word-phrase match."""
    q = _norm(query)
    qw = q.split()
    best = 0.0
    for field in ("text", "en"):
        cand = _norm(seg.get(field) or "")
        if not cand:
            continue
        if q and q in cand:
            return 1.0
        best = max(best, difflib.SequenceMatcher(None, q, cand).ratio())
        best = max(best, _word_score(qw, cand.split()))
    return best


def _resolve_file(source: str) -> Path | None:
    for p in CONFIG.recordings_dir.iterdir():
        if p.stem == source:
            return p
    return None


def find(query: str, top: int = 3) -> list[dict]:
    master = json.loads((CONFIG.out_dir / "master.json").read_text(encoding="utf-8"))
    sessions = json.loads((CONFIG.work_dir / "sessions.json").read_text(encoding="utf-8"))
    offsets_by_day = {}
    for entry in sessions if isinstance(sessions, list) else []:
        offsets_by_day[entry["label"]] = entry["offsets"]
    # sessions.json may also just be the flat offsets dict (see work/sessions.json
    # written by sessionize) keyed by device id, without day grouping — handle both.

    scored = [(_score(query, seg), seg) for seg in master]
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, seg in scored[:top]:
        if score <= 0:
            continue
        day = seg.get("day", "?")
        offsets = offsets_by_day.get(day, {})
        hits = []
        for source in seg.get("sources", []):
            off = offsets.get(source, {}).get("offset", 0.0)
            file_t = seg["start"] - off
            path = _resolve_file(source)
            hits.append({
                "source": source,
                "file": path.name if path else f"{source} (file not found)",
                "file_time": file_t,
                "file_time_hms": _hms(file_t),
            })
        results.append({
            "score": round(score, 3),
            "day": day,
            "session_time": seg["start"],
            "session_time_hms": _hms(seg["start"]),
            "text": seg.get("text"),
            "en": seg.get("en"),
            "sources": hits,
        })
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", help="quote / phrase to locate")
    ap.add_argument("--top", type=int, default=3, help="how many candidate segments to show")
    args = ap.parse_args()

    results = find(args.text, top=args.top)
    if not results:
        print("No match found.")
        return
    for r in results:
        print(f"\n[score {r['score']}] session {r['day']} @ {r['session_time_hms']}")
        print(f"  text: {r['text']}")
        if r["en"] and r["en"] != r["text"]:
            print(f"  en:   {r['en']}")
        for h in r["sources"]:
            print(f"  -> {h['file']}  @ {h['file_time_hms']}  ({h['file_time']:.1f}s)")


if __name__ == "__main__":
    main()
