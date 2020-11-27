#!/bin/bash
#SBATCH --job-name=preprocess_data_new_%j
#SBATCH --output=render_colmap_%j.out
#SBATCH --mem=32G
#SBATCH --time=0-3:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

echo "Running on $(hostname)"
echo "The $(type python)"
echo

DATASET_PATH=$(cat params.yaml | yq -r '.render_colmap.root')
PLY_FILE=$(cat params.yaml | yq -r '.render_colmap.ply_file')
WORKSPACE=/home/kremeto1/neural_rendering

echo Running:
echo python $WORKSPACE/colmap/load_data.py
echo     --src_reference=$DATASET_PATH/images
echo     --src_colmap=$DATASET_PATH/sparse
echo     --ply_path=$DATASET_PATH/$PLY_FILE
echo     --src_output=$(cat params.yaml | yq -r '.render_colmap.src_output')
echo     --val_ratio=$(cat params.yaml | yq -r '.render_colmap.val_ratio')
echo     --point_size=$(cat params.yaml | yq -r '.render_colmap.point_size')
echo     --min_size=$(cat params.yaml | yq -r '.render_colmap.min_size')
echo     --downsample=$(cat params.yaml | yq -r '.render_colmap.downsample')
echo     --verbose
echo

python $WORKSPACE/colmap/load_data.py \
    --src_reference=$DATASET_PATH/images \
    --src_colmap=$DATASET_PATH/sparse \
    --ply_path=$DATASET_PATH/$PLY_FILE \
    --src_output=$(cat params.yaml | yq -r '.render_colmap.src_output')\
    --val_ratio=$(cat params.yaml | yq -r '.render_colmap.val_ratio') \
    --point_size=$(cat params.yaml | yq -r '.render_colmap.point_size') \
    --min_size=$(cat params.yaml | yq -r '.render_colmap.min_size') \
    --downsample=$(cat params.yaml | yq -r '.render_colmap.downsample') \
    --verbose
