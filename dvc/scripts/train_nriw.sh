#!/bin/bash
#SBATCH --job-name=nriw
#SBATCH --output=logs/train_nriw_%j.log
#SBATCH --mem=64G
#SBATCH --time=4-0:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='amd-01'

set -e

. /opt/ohpc/admin/lmod/lmod/init/zsh
ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85

nvidia-smi

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

SCRIPT_DIR_NAME="$(basename $( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ))"

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Running ${SCRIPT_DIR_NAME} ${sub}"
echo

WORKSPACE=/home/kremeto1/neural_rendering

# Possibility to restart training.
if [ -z ${TIMESTAMP} ]; then
    STAMP=$(date "+%F-%H-%M-%S")
else
    STAMP="${TIMESTAMP}"
fi

if [ $(nvidia-smi -L | cut -f 3 -d' ' | head -n 1) == 'GeForce' ]; then
    PT_BS=64
    PT_STEPS=8000
    FD_BS=10
    FD_KIMG=12000
    FT_BS=10
    FT_KIMG=12000
else
    PT_BS=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_batch_size // "64"')
    PT_STEPS=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_appearance_steps // "7000"')
    FD_BS=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_batch_size // "16"')
    FD_KIMG=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_total_kimg // "4000"')
    FT_BS=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_batch_size // "16"')
    FT_KIMG=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_total_kimg // "1000"')
fi


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/pretrain_appearance.py"
echo "    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')"
echo "    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")')"
# Final datasets' subfolder contains only tfrecords, post_processed subfolder contains images.
echo "    --imageset_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_imageset_dir // (.train_nriw_'$sub'.dataset_parent_dir | sub("final"; "post_processed") | . += "/train")')"
echo "    --batch_size=${PT_BS}"
echo "    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"')"
echo "    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"')"
echo "    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"')"
echo "    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "4"')"
echo "    --metadata_output_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_metadata_output_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")')"
echo "    --vgg16_path=${WORKSPACE}/vgg16_weights/vgg16.npy"
echo "    --appearance_pretrain_steps=${PT_STEPS}"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/pretrain_appearance.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")') \
    --imageset_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_imageset_dir // (.train_nriw_'$sub'.dataset_parent_dir | sub("final"; "post_processed") | . += "/train")') \
    --batch_size=${PT_BS} \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "4"') \
    --metadata_output_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_metadata_output_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy" \
    --appearance_pretrain_steps=${PT_STEPS}


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py"
echo "    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')"
echo "    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir')"
echo "    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-fixed_appearance")')"
echo "    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_pretrained_app_encoder // "True"')"
echo "    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_pretrain_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")')"
echo "    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_app_encoder // "False"')"
echo "    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_from_another_ckpt // "False"')"
echo "    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_train_dir // ""')"
echo "    --total_kimg=${FD_KIMG}"
echo "    --batch_size=${FD_BS}"
echo "    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"')"
echo "    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"')"
echo "    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"')"
echo "    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "7"')"
echo "    --vgg16_path=${WORKSPACE}/vgg16_weights/vgg16.npy"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-fixed_appearance")') \
    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_pretrained_app_encoder // "True"') \
    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_pretrain_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-app_pretrain")') \
    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_app_encoder // "False"') \
    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_from_another_ckpt // "False"') \
    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_train_dir // ""') \
    --total_kimg=${FD_KIMG} \
    --batch_size=${FD_BS} \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "7"') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"


echo
echo "Running"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py"
echo "    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')"
echo "    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir')"
echo "    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-finetune_appearance")')"
echo "    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_pretrained_app_encoder // "False"')"
echo "    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_pretrain_dir // ""')"
echo "    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_app_encoder // "True"')"
echo "    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_from_another_ckpt // "True"')"
echo "    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_fixed_appearance_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-fixed_appearance")')"
echo "    --total_kimg=${FT_KIMG}"
echo "    --batch_size=${FT_BS}"
echo "    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"')"
echo "    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"')"
echo "    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"')"
echo "    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "7"')"
echo "    --vgg16_path=${WORKSPACE}/vgg16_weights/vgg16.npy"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/neural_rerendering.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-finetune_appearance")') \
    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_pretrained_app_encoder // "False"') \
    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_pretrain_dir // ""') \
    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_app_encoder // "True"') \
    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_from_another_ckpt // "True"') \
    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_fixed_appearance_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${STAMP}'-fixed_appearance")') \
    --total_kimg=${FT_KIMG} \
    --batch_size=${FT_BS} \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_buffer_appearance // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.use_semantic // "True"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.deep_buffer_nc // "7"') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"
