#!/bin/bash
#SBATCH --job-name=infer_nriw_on_subfolders_%j
#SBATCH --output=logs/infer_nriw_on_subfolders_%j.log
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --exclude='dgx-[2,5],amd-[01-02],node-[12]'

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
INPUT_FOLDER="$(realpath "$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.input_dir')")"
OUTPUT_FOLDER="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.output_dir')"

for subfolder in $(find "${INPUT_FOLDER}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'); do
    echo
    echo "Running:"
    echo "    python ../../neural_rerendering.py"
    echo "        --train_dir=$(realpath $(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.trained_model_dir'))"
    echo "        --run_mode='eval_dir'"
    echo "        --inference_input_path=${INPUT_FOLDER}/${subfolder}"
    echo "        --inference_output_dir=${OUTPUT_FOLDER}/${subfolder}"
    echo "        --train_resolution=$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.train_resolution // "512"')"
    echo "        --use_semantic=$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.use_semantic // "False"')"
    echo "        --use_buffer_appearance=$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.use_buffer_appearance // "False"')"
    echo "        --appearance_nc=$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.appearance_nc // "3"')"
    echo "        --deep_buffer_nc=$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.deep_buffer_nc // "4"')"
    echo

    if ls "${OUTPUT_FOLDER}/${subfolder}"
    then
        continue
    fi
    mkdir -p "${OUTPUT_FOLDER}/${subfolder}"

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python ../../neural_rerendering.py \
        --train_dir="$(realpath "$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.trained_model_dir')")" \
        --run_mode="eval_dir" \
        --inference_input_path="${INPUT_FOLDER}/${subfolder}" \
        --inference_output_dir="${OUTPUT_FOLDER}/${subfolder}" \
        --train_resolution="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.train_resolution // "512"')" \
        --use_semantic="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.use_semantic // "False"')" \
        --use_buffer_appearance="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.use_buffer_appearance // "False"')" \
        --appearance_nc="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.appearance_nc // "3"')" \
        --deep_buffer_nc="$(cat params.yaml | yq -r '.infer_nriw_on_subfolders_'$sub'.deep_buffer_nc // "4"')"

done
