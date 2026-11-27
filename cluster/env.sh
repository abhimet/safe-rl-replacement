# Source this (`source cluster/env.sh`) before running anything on Zaratan.
# Everything (venv, results, caches, temp files) stays inside $ROOT.
ROOT=/scratch/zt1/project/msai635/user/smalkir1/shielded-rl
source /etc/profile >/dev/null 2>&1
module load python >/dev/null 2>&1
unset PYTHONPATH  # the python module's site-packages would shadow the venv
source "$ROOT/.venv/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="$ROOT/.cache" MPLCONFIGDIR="$ROOT/.cache/matplotlib" TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/.cache/pip" PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
export SHIELDED_RL_RESULTS="$ROOT/results"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" "$PIP_CACHE_DIR" "$SHIELDED_RL_RESULTS" "$ROOT/logs"
cd "$ROOT"
