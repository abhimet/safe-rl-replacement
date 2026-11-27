#!/bin/bash
# Tuning round 2, part A (dev maps only, seeds 0-1): the navigation decision for highest_q@cf.
# Round-1 baseline and +conv run together (4 processes), then Phi = -d on the winner (2).
#SBATCH -J shielded-rl-tune-r2-nav
#SBATCH -A msai635-class
#SBATCH -p standard
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 1:30:00
#SBATCH -o /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/logs/tune-r2-nav-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=smalkir1@umd.edu

source /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/cluster/env.sh   # module, venv, caches and temp files inside the project folder
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD), $(python --version)"
python tune.py --round 2 --stage nav_r2 --jobs "$SLURM_CPUS_PER_TASK"
