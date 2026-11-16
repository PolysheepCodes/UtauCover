#!/bin/bash
# One-time setup for UtauCover on macOS (Apple Silicon).
# The Python environment lives in ~/Library/Application Support/UtauCover (NOT in this folder),
# because ~/Documents is synced by iCloud Drive, which makes a 20,000-file environment crawl.
set -e
cd "$(dirname "$0")"
HOME_DIR="$HOME/Library/Application Support/UtauCover"
VENV="$HOME_DIR/venv"

command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
command -v uv >/dev/null || brew install uv
command -v ffmpeg >/dev/null || brew install ffmpeg

mkdir -p "$HOME_DIR"
[ -x "$VENV/bin/python" ] || uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -e .
ln -sfn "$VENV" .venv

echo
echo "Downloading the Whisper lyrics model (1.6 GB, one time)..."
"$VENV/bin/python" -c "import whisper, os; whisper._download(whisper._MODELS['turbo'], os.path.expanduser('~/.cache/whisper'), False)"
echo
echo "Done. Separation models download on first use (~0.3-0.7 GB each)."
echo "  Window:   open UtauCover.app"
echo "  Terminal: .venv/bin/utaucover build song.mp3 --lyrics lyrics.txt --singer Googoloid"
