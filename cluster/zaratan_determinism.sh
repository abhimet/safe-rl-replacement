#!/bin/bash
# Determinism check (run after the core job): retrain 2 finished core runs from scratch on Zaratan
# and require bit-identical training CSV, checkpoint and evaluation. Writes only to $ROOT/tmp.
#SBATCH -J shielded-rl-determinism
#SBATCH -A msai635-class
#SBATCH -p standard
#SBATCH -n 1
#SBATCH -c 2
#SBATCH -t 0:30:00
#SBATCH -o /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/logs/determinism-%j.out

source /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/cluster/env.sh   # module, venv, caches and temp files inside the project folder
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD)"
python cluster/check_determinism.py --results "$SHIELDED_RL_RESULTS/core" --tier core \
    --runs highest_q@none:42 fabr_v2@margin:42 --jobs "$SLURM_CPUS_PER_TASK"
