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

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

echo "Running on $(hostname)"
echo "The $(type python)"
echo

DATASET_PATH=$(cat params.yaml | yq -r '.render_colmap_'$sub'.root')
PLY_FILE=$(cat params.yaml | yq -r '.render_colmap_'$sub'.ply_file')
WORKSPACE=/home/kremeto1/neural_rendering

echo Running:
echo time python $WORKSPACE/colmap/load_data.py
echo     --src_reference=$DATASET_PATH/images
echo     --src_colmap=$DATASET_PATH/sparse
echo     --ply_path=$DATASET_PATH/$PLY_FILE
echo     --src_output=$(cat params.yaml | yq -r '.render_colmap_'$sub'.src_output')
echo     --val_ratio=$(cat params.yaml | yq -r '.render_colmap_'$sub'.val_ratio // "0.2"')
echo     --point_size=$(cat params.yaml | yq -r '.render_colmap_'$sub'.point_size // "2.0"')
echo     --min_size=$(cat params.yaml | yq -r '.render_colmap_'$sub'.min_size // "512"')
echo     --downsample=$(cat params.yaml | yq -r '.render_colmap_'$sub'.downsample')
echo     --verbose
echo

time python $WORKSPACE/colmap/load_data.py \
    --src_reference=$DATASET_PATH/images \
    --src_colmap=$DATASET_PATH/sparse \
    --ply_path=$DATASET_PATH/$PLY_FILE \
    --src_output=$(cat params.yaml | yq -r '.render_colmap_'$sub'.src_output') \
    --val_ratio=$(cat params.yaml | yq -r '.render_colmap_'$sub'.val_ratio // "0.2"') \
    --point_size=$(cat params.yaml | yq -r '.render_colmap_'$sub'.point_size // "2.0"') \
    --min_size=$(cat params.yaml | yq -r '.render_colmap_'$sub'.min_size // "512"') \
    --downsample=$(cat params.yaml | yq -r '.render_colmap_'$sub'.downsample') \
    --verbose
