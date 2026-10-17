#!/usr/bin/env bash
set -euo pipefail
repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# Keep the caller working directory so relative data/results retain their meaning.
export PYTHONPATH="$repository_root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m tools.saer_transfer.run "$@"
