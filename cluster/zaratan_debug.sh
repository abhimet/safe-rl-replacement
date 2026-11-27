#!/bin/bash
# Setup check (debug partition): the test suite and reproduce.py smoke.
#SBATCH -J shielded-rl-debug
#SBATCH -A msai635-class
#SBATCH -p debug
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 0:15:00
#SBATCH -o /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/logs/debug-%j.out

source /scratch/zt1/project/msai635/user/smalkir1/shielded-rl/cluster/env.sh   # module, venv, caches and temp files inside the project folder
set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $SLURM_CPUS_PER_TASK CPUs, commit $(git rev-parse --short HEAD), $(python --version)"
echo "protocol sha256 $(sha256sum configs/final_protocol.json | cut -c1-64)"
python reproduce.py smoke
