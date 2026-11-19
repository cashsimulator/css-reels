#!/usr/bin/env bash
# One-time setup for a fresh session. Safe to run again.
set -e
cd "$(dirname "$0")"
pip install --break-system-packages -q kokoro-onnx soundfile opencv-python-headless pillow scipy numpy >/dev/null 2>&1 || \
  pip install -q kokoro-onnx soundfile opencv-python-headless pillow scipy numpy
which ffmpeg >/dev/null || (apt-get install -y -qq ffmpeg >/dev/null 2>&1)
M="${CSS_MODELS:-$HOME/.cache/css-models}"; mkdir -p "$M"
B=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -s "$M/voices-v1.0.bin" ]  || curl -sSL --max-time 600 -o "$M/voices-v1.0.bin"  "$B/voices-v1.0.bin"
[ -s "$M/kokoro-v1.0.onnx" ] || curl -sSL --max-time 900 -o "$M/kokoro-v1.0.onnx" "$B/kokoro-v1.0.onnx"
git config user.name  >/dev/null || git config user.name  "css-reels bot"
git config user.email >/dev/null || git config user.email "bot@cashsimulator.app"
echo "setup ok"
