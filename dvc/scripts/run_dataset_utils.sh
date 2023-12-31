#!/bin/bash
#SBATCH --job-name=run_dataset_utils
#SBATCH --output=logs/run_dataset_utils_%j.log
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16

set -e
. /opt/ohpc/admin/lmod/lmod/init/bash

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

sub=$1

nvidia-smi

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

WORKSPACE=/home/kremeto1/neural_rendering

echo
echo "Running:"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ${WORKSPACE}/dataset_utils.py"
echo "    --dataset_name=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.dataset_name')"
echo "    --dataset_parent_dir=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.dataset_parent_dir')"
echo "    --output_dir=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.output_dir')"
echo "    --xception_frozen_graph_path=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.xception_frozen_graph_path // "/home/kremeto1/neural_rendering/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb"')"
echo "    --use_semantic_gt=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.use_semantic_gt // "True"')"
echo "    --use_semantic=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.use_semantic // "True"')"
echo "    --alsologtostderr"
echo

mkdir -p $(cat params.yaml | yq -r '.dataset_utils_'$sub'.output_dir')

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ${WORKSPACE}/dataset_utils.py \
    --dataset_name=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.dataset_parent_dir') \
    --output_dir=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.output_dir') \
    --xception_frozen_graph_path=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.xception_frozen_graph_path // "/home/kremeto1/neural_rendering/deeplabv3_xception_ade20k_train/frozen_inference_graph.pb"') \
    --use_semantic_gt=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.use_semantic_gt // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.dataset_utils_'$sub'.use_semantic // "True"') \
    --run_mode="" \
    --alsologtostderr
