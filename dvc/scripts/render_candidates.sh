#!/bin/bash
#SBATCH --job-name=render_candidates
#SBATCH --output=logs/render_candidates_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='dgx-[3,5],amd-[01-02],node-[12]'

set -e

. /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo

WORKSPACE=/home/kremeto1/neural_rendering

echo
echo "Running:"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/inloc/render_candidates.py"
echo "    --src_output=$(cat params.yaml | yq -r '.render_candidates_'$sub'.src_output')"
echo "    --input_poses=$(cat params.yaml | yq -r '.render_candidates_'$sub'.input_poses')"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/inloc/render_candidates.py \
    --src_output=$(cat params.yaml | yq -r '.render_candidates_'$sub'.src_output') \
    --input_poses=$(cat params.yaml | yq -r '.render_candidates_'$sub'.input_poses')
