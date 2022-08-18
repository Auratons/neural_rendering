#!/bin/bash
#SBATCH --job-name=build_inloc_query
#SBATCH --output=logs/inloc_query_build_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4

set -euo pipefail

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

nvidia-smi

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}
export PYTHONUNBUFFERED=1  # for tqdm into file

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

WORKSPACE=/home/kremeto1/neural_rendering

echo
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python"
echo "    ${WORKSPACE}/inloc/render_inloc_query.py"
echo "        --inloc_path='/home/kremeto1/neural_rendering/datasets/raw/inloc'"
echo "        --query_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.query_path // "/home/kremeto1/neural_rendering/datasets/raw/inloc/query/with_borders"')"
echo "        --output_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.output_path')"
echo "        --mat_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.mat_path')"
echo "        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.n_max_per_scan // "None"')"
echo "        --point_size=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.point_size // "5"')"
echo "        --max_depth=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.max_depth // "20"')"
echo "        --max_img_size=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.max_img_size // "4032"')"
echo "        --nvidia_id=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.nvidia_id // "0"')"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python \
    ${WORKSPACE}/inloc/render_inloc_query.py \
        --inloc_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.inloc_path // "/home/kremeto1/neural_rendering/datasets/raw/inloc"') \
        --query_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.query_path // "/home/kremeto1/neural_rendering/datasets/raw/inloc/query/with_borders"') \
        --output_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.output_path') \
        --mat_path=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.mat_path') \
        --n_max_per_scan=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.n_max_per_scan // "None"') \
        --point_size=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.point_size // "5"') \
        --max_depth=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.max_depth // "20"') \
        --max_img_size=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.max_img_size // "4032"') \
        --nvidia_id=$(cat params.yaml | yq -r '.render_inloc_query_'$sub'.nvidia_id // "0"')

exit 0
