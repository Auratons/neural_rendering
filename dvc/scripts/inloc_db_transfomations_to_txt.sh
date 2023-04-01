#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --output=logs/inloc_txt_%j.log
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2

set -euo pipefail

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=/usr/local/bin:/usr/bin

set +u
echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo
set -u

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
    exec ~/containers/renderer-app.sif \
        ~/.conda/envs/pipeline/bin/python \
            ~/neural_rendering/inloc/inloc_db_transfomations_to_txt.py \
                --inloc_path ~/neural_rendering/datasets/raw/inloc \
                --inloc_rendered_by_pyrender ~/neural_rendering/datasets/processed/inloc/inloc_rendered_base \
                --output_path ~/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting \
                --n_max_per_scan 30000000
