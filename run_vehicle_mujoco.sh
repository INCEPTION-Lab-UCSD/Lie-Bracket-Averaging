#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python_path="$(uv python find)"
python_real_path="$("$python_path" -c 'import os, sys; print(os.path.realpath(sys.executable))')"
python_lib_dir="$(cd "$(dirname "$python_real_path")/../lib" && pwd)"

export DYLD_FALLBACK_LIBRARY_PATH="${python_lib_dir}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"

exec uv run mjpython vehicle_mujoco.py "$@"
