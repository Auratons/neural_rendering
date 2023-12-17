#!/bin/bash
#SBATCH --job-name=infer_nriw_%j
#SBATCH --output=logs/infer_nriw_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --exclude='node-[12]'

set -euo pipefail

. /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85

nvidia-smi

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(realpath "${SCRIPT_DIR}/../../")"
INPUT_FOLDER="$(realpath "$(cat params.yaml | yq -r '.infer_nriw_'$sub'.input_dir')")"
OUTPUT_FOLDER="$(cat params.yaml | yq -r '.infer_nriw_'$sub'.output_dir')"

echo
echo "Running:"
echo "    python ../../neural_rerendering.py"
echo "        --train_dir=$(realpath $(cat params.yaml | yq -r '.infer_nriw_'$sub'.trained_model_dir'))"
echo "        --run_mode='eval_dir'"
echo "        --inference_input_path=${INPUT_FOLDER}"
echo "        --inference_output_dir=${OUTPUT_FOLDER}"
echo "        --train_resolution=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.train_resolution // "512"')"
echo "        --use_buffer_appearance=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.use_buffer_appearance // "True"')"
echo "        --use_semantic=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.use_semantic // "True"')"
echo "        --appearance_nc=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.appearance_nc // "10"')"
echo "        --deep_buffer_nc=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.deep_buffer_nc // "7"')"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ../../neural_rerendering.py \
    --train_dir="$(realpath "$(cat params.yaml | yq -r '.infer_nriw_'$sub'.trained_model_dir')")" \
    --run_mode="eval_dir" \
    --inference_input_path="${INPUT_FOLDER}" \
    --inference_output_dir="${OUTPUT_FOLDER}" \
    --train_resolution="$(cat params.yaml | yq -r '.infer_nriw_'$sub'.train_resolution // "512"')" \
    --use_buffer_appearance="$(cat params.yaml | yq -r '.infer_nriw_'$sub'.use_buffer_appearance // "True"')" \
    --use_semantic=$(cat params.yaml | yq -r '.infer_nriw_'$sub'.use_semantic // "True"') \
    --appearance_nc="$(cat params.yaml | yq -r '.infer_nriw_'$sub'.appearance_nc // "10"')" \
    --deep_buffer_nc="$(cat params.yaml | yq -r '.infer_nriw_'$sub'.deep_buffer_nc // "7"')"
