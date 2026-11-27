#!/usr/bin/env bash
# One command for graders (laptop CPU, about 35-40 min in total):
#
#   ./grade.sh            # set up .venv, then: smoke tests, exact verification, quick sanity check
#   ./grade.sh --no-quick # skip the 30-min retraining step
#
#   smoke   the full test suite (~1 min)
#   verify  EXACT reproduction: re-evaluates every shipped final checkpoint (docs/results/v2/)
#           and re-aggregates; every reported number must match (~3 min)
#   quick   SANITY CHECK OF DIRECTION, not a reproduction: retrains 5 headline configs x 3 seeds
#           on this machine (~30 min). Training is bit-reproducible on one machine and software
#           stack, but not across CPUs / torch versions, so these numbers will differ from the
#           reported ones; check only that the pattern (which configs collide shield-off) holds.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  if [ "$(uname -s)" = "Linux" ]; then  # avoid the multi-GB CUDA wheels; CPU is all we need
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/python -m pip install -r requirements.txt
fi
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
steps=(smoke verify quick)
[ "${1:-}" = "--no-quick" ] && steps=(smoke verify)
.venv/bin/python reproduce.py "${steps[@]}"
