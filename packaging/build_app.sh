#!/usr/bin/env bash
# Build the trimmed macOS app (SMPL neutral + MANO) into dist/.
# Usage: packaging/build_app.sh            (uses the active python)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --quiet "pyinstaller>=6.0"
rm -rf "build/bmvis" "dist/Body Model Visualizer" "dist/Body Model Visualizer.app"
python -m PyInstaller --noconfirm --workpath build/bmvis packaging/bmvis.spec
echo "Built: dist/Body Model Visualizer.app"
du -sh "dist/Body Model Visualizer.app"
