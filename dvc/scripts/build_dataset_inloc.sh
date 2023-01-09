#!/bin/bash
#SBATCH --job-name=build_inloc_data
#SBATCH --output=logs/inloc_data_build_%j.log
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='dgx-[2,3,4,5]'

set -euo pipefail

. /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

nvidia-smi

sub=$1
output=$2

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
# export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}
export PATH=/usr/local/bin:/usr/bin:~/.conda/envs/pipeline/bin/:~/tools

set +u
echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo
set -u

WORKSPACE=/home/kremeto1/neural_rendering

echo
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python"
echo "    ${WORKSPACE}/inloc/render_inloc_db.py"
echo "        --inloc_path='/home/kremeto1/neural_rendering/datasets/raw/inloc'"
echo "        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_'$sub'.n_max_per_scan')"
echo "        --point_size=$(cat params.yaml | yq -r '.render_inloc_'$sub'.point_size')"
echo "        --bg_color=$(cat params.yaml | yq -r '.render_inloc_'$sub'.bg_color')"
echo "        --max_depth=$(cat params.yaml | yq -r '.render_inloc_'$sub'.max_depth // "-1"')"
echo "        --width=$(cat params.yaml | yq -r '.render_inloc_'$sub'.width')"
echo "        --output_path=${output}"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' \
    ~/.conda/envs/pipeline/bin/python ${WORKSPACE}/inloc/render_inloc_db.py \
        --inloc_path='/home/kremeto1/neural_rendering/datasets/raw/inloc' \
        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_'$sub'.n_max_per_scan') \
        --point_size=$(cat params.yaml | yq -r '.render_inloc_'$sub'.point_size') \
        --bg_color=$(cat params.yaml | yq -r '.render_inloc_'$sub'.bg_color') \
        --max_depth=$(cat params.yaml | yq -r '.render_inloc_'$sub'.max_depth // "-1"') \
        --width=$(cat params.yaml | yq -r '.render_inloc_'$sub'.width') \
        --output_path=${output}

exit 0
