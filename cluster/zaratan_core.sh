#!/bin/bash
# Core tier of the frozen configs/final_protocol.json: 11 configs x seeds 42-51 = 110 runs.
# Phase 1: 100 runs (4 full waves on 25 CPUs); phase 2: fabr_fixed@credit x 10 (needs every fabr@credit seed).
# Resumable: resubmitting reuses every finished run. All outputs stay under $ROOT/results/core.
#SBATCH -J shielded-rl-core
#SBATCH -A msai635-class
#SBATCH -p standard
#SBATCH -n 1
#SBATCH -c 25
#SBATCH -t 1:30:00
#SBATCH -o /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/logs/core-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smalkir1@umd.edu

source /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/cluster/env.sh   # module, venv, caches and temp files inside the project folder
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD), $(python --version)"
echo "protocol sha256 $(sha256sum configs/final_protocol.json | cut -c1-64)"
python reproduce.py core --jobs "$SLURM_CPUS_PER_TASK" --results-root "$SHIELDED_RL_RESULTS"
