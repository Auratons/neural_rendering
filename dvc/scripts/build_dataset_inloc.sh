#!/bin/bash
#SBATCH --job-name=build_inloc_data
#SBATCH --output=logs/inloc_data_build_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16

set -euo pipefail

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

nvidia-smi

sub=$1
output=$2

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}
export PYTHONUNBUFFERED=1  # for tqdm into file

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(realpath "${SCRIPT_DIR}/../../")"

echo
echo "~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python"
echo "    ${WORKSPACE}/inloc/render_inloc_db.py"
echo "        --inloc_path='/home/kremeto1/neural_rendering/datasets/raw/inloc'"
echo "        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_'$sub'.n_max_per_scan')"
echo "        --point_size=$(cat params.yaml | yq -r '.render_inloc_'$sub'.point_size')"
echo "        --bg_color=$(cat params.yaml | yq -r '.render_inloc_'$sub'.bg_color')"
echo "        --output_path=${output}"
echo

~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python \
    ${WORKSPACE}/inloc/render_inloc_db.py \
        --inloc_path='/home/kremeto1/neural_rendering/datasets/raw/inloc' \
        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_'$sub'.n_max_per_scan') \
        --point_size=$(cat params.yaml | yq -r '.render_inloc_'$sub'.point_size') \
        --bg_color=$(cat params.yaml | yq -r '.render_inloc_'$sub'.bg_color') \
        --output_path=${output}

exit 0
