#!/bin/bash
#SBATCH --job-name=cutouts
#SBATCH --output=logs/inloc_algo_cutouts_%j.log
#SBATCH --mem=8G
#SBATCH --time=0-24:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --exclude='node-[11,08,09,10]'

set -euo pipefail

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

set +u
echo
echo "Running on $(hostname)"
echo
set -u

if [[ "$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.type // "COLMAP"')" == "COLMAP" ]]; then
    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/renderer-app.sif"
    echo "        ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_colmap.py"
    echo "        --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root')"
    echo "        --input_ply_path=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_ply_path')"
    echo "        --input_root_colmap=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_colmap')"
    echo "        --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root')"
    echo "        --test_size=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.test_size')"
    echo "        --squarify=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.squarify')"
    echo "        --val_ratio=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.val_ratio // "0.2"')"
    echo "        --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --bind /nfs:/nfs ~/containers/renderer-app.sif \
            ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_colmap.py \
            --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root') \
            --input_ply_path=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_ply_path') \
            --input_root_colmap=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_colmap') \
            --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root') \
            --test_size=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.test_size') \
            --squarify=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.squarify') \
            --val_ratio=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.val_ratio // "0.2"') \
            --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')
fi


if [[ "$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.type // "COLMAP"')" == "ARTWIN" ]]; then
    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/renderer-app.sif"
    echo "        ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_artwin.py"
    echo "        --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root')"
    echo "        --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root')"
    echo "        --input_mapping=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_mapping')"
    echo "        --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --bind /nfs:/nfs ~/containers/renderer-app.sif \
            ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_artwin.py \
            --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root') \
            --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root') \
            --input_mapping=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_mapping') \
            --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')
fi


if [[ "$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.type // "COLMAP"')" == "INLOC" ]]; then
    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/renderer-app.sif"
    echo "        ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_inloc.py"
    echo "        --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root')"
    echo "        --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root')"
    echo "        --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')"

    if [[ "$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.lift // "false"')" == "true" ]]; then
        echo "        --lift"
    fi
    echo

    if [[ "$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.lift // "false"')" == "true" ]]; then
        ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
            exec --bind /nfs:/nfs ~/containers/renderer-app.sif \
                ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_inloc.py \
                --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root') \
                --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root') \
                --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"') \
                --lift
    else
        ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
            exec --bind /nfs:/nfs ~/containers/renderer-app.sif \
                ~/.conda/envs/pipeline/bin/python ~/inloc/inLocCIIRC_dataset/buildCutouts/build_cutouts_inloc.py \
                --input_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root') \
                --output_root=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.output_root') \
                --input_root_renderer=$(cat params.yaml | yq -r '.inloc_cutouts_'$sub'.input_root_renderer // "pyrender"')
    fi
fi

exit 0
