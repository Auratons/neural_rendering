#!/bin/bash
#SBATCH --job-name=evaluate_nriw
#SBATCH --output=logs/evaluate_nriw_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --exclude='amd-[01,02],node-[12]'

set -e
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

if [ -z ${TIMESTAMP} ]; then
    TIMESTAMP=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.timestamp')
fi

WORKSPACE=/home/kremeto1/neural_rendering
SUFFIX=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.suffix // ""')
VIRTUAL_SEQ_NAME=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.virtual_seq_name // "val"')
TRAIN_DIR=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.train_dir // (.evaluate_nriw_'$sub'.model_parent_dir + .evaluate_nriw_'$sub'.dataset_name + "-'${TIMESTAMP}'-finetune_appearance'${SUFFIX}'/")')

echo
echo "Evaluating the validation set"
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py"
echo "   --dataset_name=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.dataset_name')"
echo "   --dataset_parent_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.dataset_parent_dir')"
echo "   --train_dir=${TRAIN_DIR}"
echo "   --run_mode='eval_subset'"
echo "   --virtual_seq_name=${VIRTUAL_SEQ_NAME}"
echo "   --output_validation_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.output_validation_dir // (.evaluate_nriw_'$sub'.model_parent_dir + "../evaluations/" + .evaluate_nriw_'$sub'.dataset_name + "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}''${SUFFIX}'")')"
echo "   --use_buffer_appearance=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_buffer_appearance // "True"')"
echo "   --use_semantic_gt=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_semantic_gt // "True"')"
echo "   --use_semantic=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_semantic // "True"')"
echo "   --appearance_nc=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.appearance_nc // "10"')"
echo "   --deep_buffer_nc=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.deep_buffer_nc // "7"')"
echo "   --logtostderr"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py \
    --dataset_name=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.dataset_parent_dir') \
    --train_dir=${TRAIN_DIR} \
    --run_mode='eval_subset' \
    --virtual_seq_name=${VIRTUAL_SEQ_NAME} \
    --output_validation_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.output_validation_dir // (.evaluate_nriw_'$sub'.model_parent_dir + "../evaluations/" + .evaluate_nriw_'$sub'.dataset_name + "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}''${SUFFIX}'")') \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_buffer_appearance // "True"') \
    --use_semantic_gt=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_semantic_gt // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.use_semantic // "True"') \
    --appearance_nc=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.deep_buffer_nc // "7"') \
    --logtostderr


echo
echo "Evaluate quantitative metrics"
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/evaluate_quantitative_metrics.py"
echo "   --val_set_out_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.output_validation_dir // (.evaluate_nriw_'$sub'.model_parent_dir + "../evaluations/" + .evaluate_nriw_'$sub'.dataset_name + "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}''${SUFFIX}'")')"
echo "   --experiment_title=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.experiment_title // (.evaluate_nriw_'$sub'.dataset_name | . += "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}'")')"
echo "   --logtostderr"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/evaluate_quantitative_metrics.py \
    --val_set_out_dir=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.output_validation_dir // (.evaluate_nriw_'$sub'.model_parent_dir + "../evaluations/" + .evaluate_nriw_'$sub'.dataset_name + "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}''${SUFFIX}'")') \
    --experiment_title=$(cat params.yaml | yq -r '.evaluate_nriw_'$sub'.experiment_title // (.evaluate_nriw_'$sub'.dataset_name | . += "_'${TIMESTAMP}'_seq-'${VIRTUAL_SEQ_NAME}'")') \
    --logtostderr
