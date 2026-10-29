#!/bin/bash
# EWORLD — build the native pipeline if needed, encode input/ into out/, and
# paint the survivors with their boxes into output/preview.mp4.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/usr/bin/python3}"   # only for the one-time .pt -> .onnx export
BUILD="$ROOT/build"

# EHC and HaloCore: $EHC_ROOT / $HALOCORE_ROOT, else siblings, else the lab tree.
find_repo() {   # find_repo <env value> <name> <marker> <env name>
  local c
  for c in "$1" "$ROOT/../$2" "$HOME/Quantum_Computing_Lab/$2"; do
    if [[ -n "$c" && -e "$c/$3" ]]; then (cd "$c" && pwd); return; fi
  done
  echo "[eworld] cannot find $2 (set $4)" >&2
  exit 1
}
EHC="$(find_repo "${EHC_ROOT:-}" EHC include/ehc/media/streaming_video_encoder.hpp EHC_ROOT)"
HALOCORE="$(find_repo "${HALOCORE_ROOT:-}" HaloCore cpp/src/doc/door_abi.h HALOCORE_ROOT)"

smoke=0
extra=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    smoke=1
  else
    extra+=("$arg")
  fi
done

if [[ ! -x "$BUILD/aerial_encode" || ! -x "$BUILD/aerial_paint" ]]; then
  if [[ ! -f "$EHC/build/libehc_core.a" ]]; then
    echo "[eworld] building EHC ehc_core in $EHC/build"
    cmake -S "$EHC" -B "$EHC/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$EHC/build" --target ehc_core -j
  fi
  echo "[eworld] configuring $BUILD"
  cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DEHC_ROOT="$EHC" -DHALOCORE_ROOT="$HALOCORE"
  cmake --build "$BUILD" -j
fi

export HALOCORE_FLORENCE_DIR="${HALOCORE_FLORENCE_DIR:-/opt/mpl/models/florence-2-base-onnx}"
export HALOCORE_FIGURES_QUANT="${HALOCORE_FIGURES_QUANT:-int8}"
mkdir -p "$ROOT/input" "$ROOT/out" "$ROOT/output" "$ROOT/models"
cd "$ROOT"

# Weights: download the Ultralytics checkpoints, export each head to ONNX once.
fetch() {
  [[ -s "$ROOT/models/$1" ]] && return
  echo "[eworld] downloading $1"
  curl -fL --retry 3 -o "$ROOT/models/$1" "https://github.com/ultralytics/assets/releases/download/v8.3.0/$1"
}
export_head() {   # export_head <pt> <onnx> [classes]
  [[ -s "$ROOT/models/$2" ]] && return
  fetch "$1"
  echo "[eworld] exporting models/$2 (one time; needs ultralytics + onnx${3:+ + CLIP})"
  if [[ -n "${3:-}" ]]; then
    "$PY" "$ROOT/tools/export_onnx.py" "$ROOT/models/$1" --classes "$3" --out "$ROOT/models/$2"
  else
    "$PY" "$ROOT/tools/export_onnx.py" "$ROOT/models/$1" --out "$ROOT/models/$2"
  fi
}
# The object layer (config/object_layer.json) names both heads; the World
# prompts are read from it in file order, so the graph and the config agree.
export_head yolov8x.pt yolov8x.onnx
world_prompts="$("$PY" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["world"]["prompts"]))' \
  "$ROOT/config/object_layer.json")"
export_head yolov8x-worldv2.pt yolov8x-worldv2-aerial.onnx "$world_prompts"

shopt -s nullglob
input=("$ROOT/input"/*.{mp4,mov,mkv,webm,avi,m4v,MP4,MOV,MKV})
shopt -u nullglob

if [[ ${#input[@]} -eq 0 && "$smoke" -eq 1 ]]; then
  echo "[eworld] input/ is empty — writing a 4s stencil plate"
  ffmpeg -y -nostdin -loglevel error \
    -f lavfi -i "color=c=0x6b7280:s=1280x720:d=4:r=10" \
    -vf "drawbox=x=180:y=220:w=520:h=280:color=0x1f2937:t=fill,drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:text='B127':fontsize=96:fontcolor=white:x=300:y=320" \
    -pix_fmt yuv420p "$ROOT/input/stencil_plate.mp4"
  input=("$ROOT/input/stencil_plate.mp4")
fi

if [[ ${#input[@]} -eq 0 ]]; then
  echo "[eworld] build is ready. Put videos in $ROOT/input and run ./run.sh again."
  echo "[eworld] ./run.sh --smoke encodes a 4 second stencil plate named B127."
  exit 0
fi

echo "[eworld] encoding ${#input[@]} clip(s)"
encode_args=(--fps 2 --frame 640x640 --objects "$ROOT/config/object_layer.json" --ocr florence)
if [[ ${#extra[@]} -gt 0 ]]; then
  encode_args+=("${extra[@]}")
fi
"$BUILD/aerial_encode" "$ROOT/input" "$ROOT/out" "${encode_args[@]}"

"$BUILD/aerial_paint" "$ROOT/out" "$ROOT/output/preview.mp4"
echo "[eworld] preview: $ROOT/output/preview.mp4"
