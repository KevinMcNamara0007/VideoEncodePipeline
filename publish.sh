#!/bin/bash
# ------------------------------------------------------------------------
# Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
# Architect: Kevin McNamara, CTO and Chief Science Officer.
# PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
# applications. No permission to copy, modify or reuse; use is strictly
# governed by the EULA (EULA.md at the repository root).
# ------------------------------------------------------------------------
# EWORLD -> the appliance. Encode the clip's triples through the HaloCore
# ingest front door of the RUNNING appliance (POST /v1/ingest = ehc::ops::ingest,
# the one call `halocore ingest` makes), which writes the native store under
# the appliance's ingest root (<home>/data/stores/<name>) and mounts it as
# the pool <name>; then put the preview and its lookups beside the shards
# (<store>/media/: preview.mp4, preview.index.json, frames.lsh, objects.jsonl,
# overlay.jsonl, frames.jsonl, videos.jsonl, config.json, triples.jsonl),
# check the store's serving relics, confirm the mount, and ask one question.
#
#   ./publish.sh <name> [--out out] [--preview output/preview.mp4]
#                       [--home /opt/mpl] [--replace] [--ask "question"]
#
# --replace unmounts and removes an existing store of that name first (the
# store is this clip's encode; nothing else lives there). The appliance's
# start.sh remounts every store under the ingest root at boot.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
NAME=""; OUT="$ROOT/out"; PREVIEW="$ROOT/output/preview.mp4"; MPL="${MPL_HOME:-/opt/mpl}"
REPLACE=0; ASK="What is the story setting of frame 26?"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --preview) PREVIEW="$2"; shift 2 ;;
    --home) MPL="$2"; shift 2 ;;
    --replace) REPLACE=1; shift ;;
    --ask) ASK="$2"; shift 2 ;;
    -h|--help) sed -n '9,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [[ -z "$NAME" ]] && NAME="$1" || { echo "one name only" >&2; exit 2; }; shift ;;
  esac
done
[[ -n "$NAME" ]] || { echo "usage: publish.sh <name> [--out DIR] [--preview MP4] [--home DIR] [--replace]" >&2; exit 2; }
[[ "$NAME" =~ ^[a-z0-9_]+$ ]] || { echo "name must be [a-z0-9_]+ (it is the pool name)" >&2; exit 2; }
OUT="$(cd "$OUT" && pwd)"
[[ -s "$OUT/triples.jsonl" ]] || { echo "no $OUT/triples.jsonl (run ./run.sh first)" >&2; exit 1; }
[[ -s "$PREVIEW" ]] || { echo "no $PREVIEW (run ./run.sh first)" >&2; exit 1; }
PREVIEW="$(cd "$(dirname "$PREVIEW")" && pwd)/$(basename "$PREVIEW")"
INDEX="${PREVIEW%.*}.index.json"
[[ -s "$INDEX" ]] || { echo "no $INDEX beside the preview (aerial_paint writes it)" >&2; exit 1; }

CONFIG="$MPL/config.env"
cfg() {   # live env first, then config.env (as start.sh)
  local v; v="$(printenv "$1" 2>/dev/null || true)"
  if [[ -n "$v" ]]; then printf '%s' "$v"; return; fi
  [[ -f "$CONFIG" ]] || return 0
  grep -E "^$1=" "$CONFIG" 2>/dev/null | head -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*//;s/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//' || true
}
PORT="$(cfg HALOAGENT_NATIVE_PORT)"; PORT="${PORT:-8099}"
URL="http://127.0.0.1:$PORT"
TOKEN="$(cfg HALOAGENT_WRITE_TOKEN)"
INGEST_ROOT="$(cfg HALOAGENT_INGEST_ROOT)"; INGEST_ROOT="${INGEST_ROOT:-$MPL/data/stores}"
STORE="$INGEST_ROOT/$NAME"
AUTH=(); [[ -n "$TOKEN" ]] && AUTH=(-H "Authorization: Bearer $TOKEN")
HALOCORE="$MPL/bin/halocore"; [[ -x "$HALOCORE" ]] || HALOCORE="$(command -v halocore || true)"

say() { printf '[publish] %s\n' "$*"; }
die() { printf '[publish] %s\n' "$*" >&2; exit 1; }
jget() { python3 -c 'import json,sys
d=json.load(sys.stdin)
for k in sys.argv[1:]:
    d=d.get(k,"") if isinstance(d,dict) else ""
print(d if not isinstance(d,(dict,list)) else json.dumps(d))' "$@"; }

curl -sf --max-time 5 "$URL/health" >/dev/null || die "appliance not up at $URL (run $MPL/start.sh)"

# ── 1. an existing store of this name ─────────────────────────────────────
if [[ -e "$STORE" ]]; then
  [[ "$REPLACE" -eq 1 ]] || die "$STORE exists; --replace to re-encode it, or choose another name"
  [[ -f "$STORE/manifest.json" || -f "$STORE/shard0/manifest.json" ]] || die "$STORE is not a store; not removing it"
  say "replacing $STORE"
  curl -s --max-time 60 -X DELETE "$URL/v1/pools" "${AUTH[@]}" -H 'Content-Type: application/json' \
    -d "{\"name\":\"$NAME\",\"purge\":true}" >/dev/null || true
  rm -rf "$STORE"
fi

# ── 2. the HaloCore ingest front door ─────────────────────────────────────
say "ingest $OUT/triples.jsonl -> $STORE (POST $URL/v1/ingest)"
RES="$(curl -s --max-time 600 -X POST "$URL/v1/ingest" "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"source\":\"$OUT/triples.jsonl\",\"store\":\"$NAME\"}")"
[[ "$(printf '%s' "$RES" | jget ok)" == "True" ]] || die "ingest failed: $RES"
say "ingested $(printf '%s' "$RES" | jget records) records, mounted=$(printf '%s' "$RES" | jget mounted)"
[[ -f "$STORE/manifest.json" ]] || die "no store at $STORE after the ingest"

# ── 3. the preview and its lookups, beside the shards ─────────────────────
mkdir -p "$STORE/media"
cp -f "$PREVIEW" "$STORE/media/preview.mp4"
cp -f "$INDEX" "$STORE/media/preview.index.json"
for f in frames.lsh objects.jsonl overlay.jsonl frames.jsonl videos.jsonl config.json encode_summary.json triples.jsonl; do
  [[ -f "$OUT/$f" ]] && cp -f "$OUT/$f" "$STORE/media/$f"
done
say "sidecars: $(ls "$STORE/media" | tr '\n' ' ')"

# ── 4. relics ─────────────────────────────────────────────────────────────
if [[ -x "$HALOCORE" ]]; then
  "$HALOCORE" relics --index "$STORE" --contract --json 2>/dev/null | python3 -c '
import json,sys
raw=sys.stdin.read().strip()
try: d=json.loads(raw.splitlines()[-1])
except Exception: print("[publish] relics: (no json)", raw[:300]); sys.exit(0)
rel=d.get("relics") or d.get("items") or []
if isinstance(rel,dict): rel=[dict(name=k,**(v if isinstance(v,dict) else {"state":v})) for k,v in rel.items()]
for r in rel:
    print("[publish] relic %-20s %s" % (r.get("name",""), r.get("state", r.get("status",""))))
c=d.get("contract")
if c: print("[publish] contract:", json.dumps(c)[:400])
' || true
  for f in subjects.tsv surfaces.ehmap sim_subjects.ehsim sim_relations.ehsim relation_surfaces.json subj_index.bin; do
    [[ -s "$STORE/$f" ]] && say "  $f $(wc -c < "$STORE/$f" | tr -d ' ') bytes"
  done
else
  say "halocore not found; relics not listed"
fi

# ── 5. mounted? served? ───────────────────────────────────────────────────
MOUNT="$(curl -s --max-time 10 "$URL/v1/lakes" "${AUTH[@]}" | python3 -c '
import json,sys
d=json.load(sys.stdin); n=sys.argv[1]
rows=[p for p in d.get("pools",[]) if p.get("name")==n]+[p for p in d.get("lakes",[]) if p.get("name")==n]
print(" ".join("%s:%s:%s" % (p.get("kind"),p.get("state"),p.get("path")) for p in rows) or "NOT IN REGISTRY")' "$NAME")"
say "registry: $MOUNT"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/v1/pools/$NAME/files/media/preview.index.json")"
say "GET /v1/pools/$NAME/files/media/preview.index.json -> $code"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -H 'Range: bytes=0-1023' "$URL/v1/pools/$NAME/files/media/preview.mp4")"
say "GET /v1/pools/$NAME/files/media/preview.mp4 (Range) -> $code"

# ── 6. one question through the query front door ──────────────────────────
ANS="$(curl -s --max-time 60 -X POST "$URL/v1/chat" -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"query": sys.argv[1]}))' "$ASK")")"
say "ask: $ASK"
printf '%s' "$ANS" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("[publish]   gate=%s text=%r" % (d.get("gate"), (d.get("text") or "")[:120]))
for f in (d.get("facts") or [])[:3]:
    print("[publish]   fact %s -- %s -> %s  [%s]" % (f.get("s"), f.get("r"), f.get("o"), f.get("expert")))'
say "done: pool '$NAME' at $STORE; ask the page at $URL"
