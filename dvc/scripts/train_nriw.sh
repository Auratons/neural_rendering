#!/bin/bash
#SBATCH --job-name=train_nriw
#SBATCH --output=train_nriw_'$sub'%j.log
#SBATCH --mem=128G
#SBATCH --time=3-0:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:Volta100:4

set -e

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85

nvidia-smi

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

echo "Running on $(hostname)"
echo "The $(type python)"
echo

WORKSPACE=/home/kremeto1/neural_rendering

# Possibility to restart training.
if [ -z ${TIMESTAMP} ]; then
    TIMESTAMP=$(date "+%F-%H-%M-%S")
fi

echo Running
echo time python $WORKSPACE/pretrain_appearance.py
echo     --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')
echo     --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")')
# Final datasets' subfolder contains only tfrecords, post_processed subfolder contains images.
echo     --imageset_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_imageset_dir // (.train_nriw_'$sub'.dataset_parent_dir | sub("final"; "post_processed") | . += "/train")')
echo     --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_buffer_appearance // "True"')
echo     --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"')
echo     --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic // "True"')
echo     --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_batch_size // "64"')
echo     --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_appearance_nc // "10"')
echo     --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_deep_buffer_nc // "4"')
echo     --metadata_output_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_metadata_output_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")')
echo     --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"
echo

time python $WORKSPACE/pretrain_appearance.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")') \
    --imageset_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_imageset_dir // (.train_nriw_'$sub'.dataset_parent_dir | sub("final"; "post_processed") | . += "/train")') \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_buffer_appearance // "True"') \
    --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic // "True"') \
    --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_batch_size // "64"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_deep_buffer_nc // "4"') \
    --metadata_output_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_metadata_output_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"


echo Running
echo time python $WORKSPACE/neural_rerendering.py
echo     --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')
echo     --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir')
echo     --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-fixed_appearance")')
echo     --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_pretrained_app_encoder // "True"')
echo     --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_pretrain_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")')
echo     --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_app_encoder // "False"')
echo     --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_from_another_ckpt // "False"')
echo     --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_fixed_appearance_train_dir // ""')
echo     --total_kimg=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_total_kimg // "400"')
echo     --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_use_buffer_appearance // "True"')
echo     --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"')
echo     --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_use_semantic // "True"')
echo     --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_batch_size // "16"')
echo     --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_nc // "10"')
echo     --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_deep_buffer_nc // "7"')
echo     --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"
echo

time python $WORKSPACE/neural_rerendering.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-fixed_appearance")') \
    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_pretrained_app_encoder // "True"') \
    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_pretrain_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-app_pretrain")') \
    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_train_app_encoder // "False"') \
    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_load_from_another_ckpt // "False"') \
    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_fixed_appearance_train_dir // ""') \
    --total_kimg=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_total_kimg // "400"') \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_use_buffer_appearance // "True"') \
    --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_use_semantic // "True"') \
    --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_batch_size // "16"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.fixed_deep_buffer_nc // "7"') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"


echo Running
echo time python $WORKSPACE/neural_rerendering.py
echo     --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name')
echo     --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir')
echo     --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-finetune_appearance")')
echo     --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_pretrained_app_encoder // "False"')
echo     --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_pretrain_dir // ""')
echo     --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_app_encoder // "True"')
echo     --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_from_another_ckpt // "True"')
echo     --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_fixed_appearance_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-fixed_appearance")')
echo     --total_kimg=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_total_kimg // "100"')
echo     --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_use_buffer_appearance // "True"')
echo     --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"')
echo     --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_use_semantic // "True"')
echo     --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_batch_size // "16"')
echo     --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_nc // "10"')
echo     --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_deep_buffer_nc // "7"')
echo     --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"
echo

time python $WORKSPACE/neural_rerendering.py \
    --dataset_name=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_name') \
    --dataset_parent_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.dataset_parent_dir') \
    --train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-finetune_appearance")') \
    --load_pretrained_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_pretrained_app_encoder // "False"') \
    --appearance_pretrain_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_pretrain_dir // ""') \
    --train_app_encoder=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_train_app_encoder // "True"') \
    --load_from_another_ckpt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_load_from_another_ckpt // "True"') \
    --fixed_appearance_train_dir=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_fixed_appearance_train_dir // (.train_nriw_'$sub'.model_parent_dir + .train_nriw_'$sub'.dataset_name | . += "-" | . += "'${TIMESTAMP}'-fixed_appearance")') \
    --total_kimg=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_total_kimg // "100"') \
    --use_buffer_appearance=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_use_buffer_appearance // "True"') \
    --use_semantic_gt=$(cat params.yaml | yq -r '.train_nriw_'$sub'.pretrain_use_semantic_gt // "True"') \
    --use_semantic=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_use_semantic // "True"') \
    --batch_size=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_batch_size // "16"') \
    --appearance_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_appearance_nc // "10"') \
    --deep_buffer_nc=$(cat params.yaml | yq -r '.train_nriw_'$sub'.finetune_deep_buffer_nc // "7"') \
    --vgg16_path="${WORKSPACE}/vgg16_weights/vgg16.npy"
