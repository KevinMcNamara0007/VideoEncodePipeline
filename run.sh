#!/bin/bash
# EWORLD — build the EHC Python bindings if needed, then encode input/ into out/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EHC="$(cd "$ROOT/../EHC" && pwd)"
PY="/usr/bin/python3"
BUILD="$EHC/build-py"
MODDIR="$BUILD/bindings/python"

smoke=0
extra=()
for arg in "$@"; do
  if [[ "$arg" == "--smoke" ]]; then
    smoke=1
  else
    extra+=("$arg")
  fi
done

need_build=0
if ! compgen -G "$MODDIR"/ehc*.so >/dev/null; then
  need_build=1
fi
if [[ ! -x "$BUILD/tools/ehc_video_concept_query" ]]; then
  need_build=1
fi

if [[ "$need_build" -eq 1 ]]; then
  echo "[eworld] configuring $BUILD"
  cmake -S "$EHC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DEHC_BUILD_PYTHON=ON \
    -DEHC_BUILD_TOOLS=ON \
    -DPython_EXECUTABLE="$PY"
  echo "[eworld] building ehc_python and ehc_video_concept_query"
  cmake --build "$BUILD" --target ehc_python ehc_video_concept_query -j
fi

export EHC_PYTHON_PATH="$MODDIR"
# The door's Florence-2 ONNX runtime and pdfium live next to halocore.
export DYLD_LIBRARY_PATH="/opt/bin${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export HALOCORE_FLORENCE_DIR="${HALOCORE_FLORENCE_DIR:-/opt/mpl/models/florence-2-base-onnx}"
export HALOCORE_FIGURES_QUANT="${HALOCORE_FIGURES_QUANT:-int8}"
mkdir -p "$ROOT/input" "$ROOT/out" "$ROOT/models"
cd "$ROOT"

MODEL="$ROOT/models/yolov8n.pt"
if [[ ! -s "$MODEL" ]]; then
  echo "[eworld] downloading yolov8n.pt"
  curl -fL --retry 3 -o "$MODEL" \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
fi

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
  echo "[eworld] bindings are ready. Put videos in $ROOT/input and run ./run.sh again."
  echo "[eworld] ./run.sh --smoke encodes a 4 second stencil plate named B127."
  exit 0
fi

echo "[eworld] encoding ${#input[@]} clip(s)"
encode_args=(--fps 2 --frame 640x640 --model "$MODEL" --ocr tesseract)
if [[ ${#extra[@]} -gt 0 ]]; then
  encode_args+=("${extra[@]}")
fi
"$PY" -u "$ROOT/aerial_overlay_encode.py" "$ROOT/input" "$ROOT/out" "${encode_args[@]}"

"$PY" -u "$ROOT/aerial_overlay_paint.py" "$ROOT/out" "$ROOT/out/preview.mp4"
echo "[eworld] preview: $ROOT/out/preview.mp4"
