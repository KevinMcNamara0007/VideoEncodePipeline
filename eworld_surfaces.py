#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
"""Publish determiner surfaces for a name-keyed HaloCore store.

HaloCore resolves an exact surface to its subject before similarity
(surface_exact). The articles a, an, and the are ordinary English
question words, the same three the retired strip_leading_determiners
path treated as a span. Another lake can already spell one of those
phrases ("the airplane" is a wiki label). When it does, that longer
surface takes the span and this store is never asked.

A name-keyed store therefore publishes those three spellings for every
subject it holds. aliases.jsonl is the encode input HaloCore already
compiles into surfaces.ehmap (subjects.tsv + name relations +
aliases.jsonl). The file is derived from subjects.tsv after ingest.
Nothing in it is a hand-picked entity.

    python3 eworld_surfaces.py /opt/mpldatalake/stores/eworld

``relics --plan --apply`` reuses a current surface map and does not
read a new aliases.jsonl. This step uses ``relics --emit``, then
remounts the pool so the running appliance mmaps the new map.
"""

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


HALOCORE = Path("/opt/bin/halocore")
LEXICON_CWD = Path("/opt/mpl")
CONFIG_ENV = Path("/opt/config/config.env")
# Articles only. The rest of query_stopwords.json is scaffolding
# (about, of, what) and must not become a subject surface.
ARTICLES = ("a", "an", "the")


def native_port() -> int:
    if not CONFIG_ENV.is_file():
        return 8099
    for line in CONFIG_ENV.read_text().splitlines():
        if line.startswith("HALOAGENT_NATIVE_PORT="):
            value = line.split("=", 1)[1].split("#", 1)[0].strip()
            return int(value or "8099")
    return 8099


def load_claims(path: Path) -> dict:
    """Map (subject, relation) to the set of objects a lake already asserts."""
    claims = {}
    if not path.is_file():
        return claims
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get("subject") or "", row.get("relation") or "")
        claims.setdefault(key, set()).add(row.get("object") or "")
    return claims


def contradicts(claims: dict, subject: str, relation: str, obj: str) -> bool:
    """True when another lake already holds this subject and relation for a different object.

    The same triple is fine. A different object makes the Ask box pick one
    lake's fact and hide the other. Callers skip the new triple.
    """
    held = claims.get((subject, relation))
    return bool(held) and obj not in held


def article_surfaces(name: str) -> list:
    folded = name.casefold().strip()
    if not folded:
        return []
    if any(folded.startswith(article + " ") for article in ARTICLES):
        return []
    return ["%s %s" % (article, folded) for article in ARTICLES]


def write_determiner_aliases(store_dir: Path) -> int:
    """Rewrite aliases.jsonl from subjects.tsv. Returns the number of surfaces."""
    subjects_path = store_dir / "subjects.tsv"
    if not subjects_path.is_file():
        raise SystemExit("no subjects.tsv in %s" % store_dir)
    names = []
    for line in subjects_path.read_text().splitlines():
        if not line.strip():
            continue
        names.append(line.split("\t", 1)[0])
    folded_owner = {}
    for name in names:
        folded_owner.setdefault(name.casefold(), name)
    rows = []
    seen = set()
    for name in names:
        for surface in article_surfaces(name):
            owner = folded_owner.get(surface)
            if owner is not None and owner != name:
                continue
            key = (surface, name)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"surface": surface, "subject": name})
    rows.sort(key=lambda row: (row["surface"], row["subject"]))
    dest = store_dir / "aliases.jsonl"
    dest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return len(rows)


def emit_surfaces(store_dir: Path) -> None:
    """Compile aliases.jsonl into surfaces.ehmap. cwd must see the relation lexicon."""
    if not HALOCORE.is_file():
        raise SystemExit("halocore not found: %s" % HALOCORE)
    subprocess.run(
        [str(HALOCORE), "relics", "--index", str(store_dir), "--emit"],
        cwd=str(LEXICON_CWD),
        check=True,
    )


def remount_store(store_dir: Path, port: int) -> dict:
    """Drop the in-process mmap and open the store again. --emit detaches it."""
    name = store_dir.name
    url = "http://127.0.0.1:%d/v1/pools" % port
    for pool in (name + "0", name):
        body = json.dumps({"name": pool}).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
        except urllib.error.HTTPError:
            pass
    body = json.dumps({
        "name": name,
        "kind": "halocore_lake",
        "path": str(store_dir),
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise SystemExit("remount failed (%s): %s" % (exc.code, detail))
    return json.loads(raw) if raw else {}


def publish_surfaces(store_dir: Path, port: int, remount: bool = True) -> int:
    count = write_determiner_aliases(store_dir)
    emit_surfaces(store_dir)
    if remount:
        remount_store(store_dir, port)
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store_dir", type=Path)
    ap.add_argument("--no-remount", action="store_true")
    args = ap.parse_args()
    store_dir = args.store_dir.resolve()
    if not store_dir.is_dir():
        raise SystemExit("store not found: %s" % store_dir)
    count = publish_surfaces(store_dir, native_port(), remount=not args.no_remount)
    print("[eworld] %d determiner surfaces in %s" % (count, store_dir / "aliases.jsonl"))


if __name__ == "__main__":
    main()
