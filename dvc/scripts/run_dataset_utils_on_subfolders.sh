#!/bin/bash
#SBATCH --job-name=run_dataset_utils_subfolders_%j
#SBATCH --output=logs/run_dataset_utils_subfolders_%j.log
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16

# Meant for usage on grouped query data (folders per
# query photo with data for candidate pose renders).

set -euo pipefail

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

sub=$1

nvidia-smi

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(realpath "${SCRIPT_DIR}/../../")"
INPUT_FOLDER="$(realpath "$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.input_dir')")"
OUTPUT_FOLDER="$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.output_dir')"

for subfolder in $(ls "${INPUT_FOLDER}"); do

    echo
    echo "Running:"
    echo "~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ${WORKSPACE}/dataset_utils.py"
    echo "    --dataset_name=${subfolder}}"
    echo "    --dataset_parent_dir=${INPUT_FOLDER}/${subfolder}"
    echo "    --output_dir=${OUTPUT_FOLDER}/${subfolder}"
    echo "    --xception_frozen_graph_path=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.xception_frozen_graph_path // "/home/kremeto1/neural_rendering/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb"')"
    echo "    --use_semantic_gt=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.use_semantic_gt // "True"')"
    echo "    --use_semantic=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.use_semantic // "True"')"
    echo "    --run_mode='eval_dir'"
    echo "    --alsologtostderr"
    echo

    mkdir -p "${OUTPUT_FOLDER}/${subfolder}"

    ~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ${WORKSPACE}/dataset_utils.py \
        --dataset_name="${subfolder}" \
        --dataset_parent_dir="${INPUT_FOLDER}/${subfolder}" \
        --output_dir="${OUTPUT_FOLDER}/${subfolder}" \
        --xception_frozen_graph_path=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.xception_frozen_graph_path // "/home/kremeto1/neural_rendering/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb"') \
        --use_semantic_gt=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.use_semantic_gt // "True"') \
        --use_semantic=$(cat params.yaml | yq -r '.dataset_utils_subfolders_'$sub'.use_semantic // "True"') \
        --run_mode="eval_dir" \
        --alsologtostderr

done
