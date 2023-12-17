#!/usr/bin/env bash
#SBATCH --job-name=cpp_render
#SBATCH --output=logs/cpp_render_%j.log
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='amd-[01,02],dgx-[4,3],node-[12,14,17,15]'

set -eu

nvidia-smi

export PATH=/usr/local/bin:/usr/bin:~/.conda/envs/pipeline/bin/:~/tools

set +u
echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo
set -u

sub="$1"

EXECUTABLE=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.renderer_executable')
ROOT_TO_PROCESS=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.root_to_process' | sed 's:/*$::')
MAX_RADIUS=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.max_radius')
OUTPUT_ROOT=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.output_root // ""')

if [[ "$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.type')" == "INLOC" ]]; then
    PLY_GLOB=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.ply_glob')
    for ply_file in `find "${ROOT_TO_PROCESS}" -type f -name "${PLY_GLOB}"`
    do
        if [[ ! -f "$(dirname ${ply_file})_SPLAT.LOCK" ]]; then
            touch "$(dirname ${ply_file})_SPLAT.LOCK"
        else
            echo "Already locked $(dirname ${ply_file})_SPLAT.LOCK, skipping."
            continue
        fi

        if [[ -z "${OUTPUT_ROOT}" ]]; then
            OUTPUT_ROOT_FOLDER=$(dirname $(dirname ${ply_file}))
        else
            OUTPUT_ROOT_FOLDER="${OUTPUT_ROOT}"
        fi

        echo
        echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
        echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
        echo "    --file ${ply_file}"
        echo "    --matrices $(dirname ${ply_file})/matrices_for_rendering.txt"
        echo "    --output_path ${OUTPUT_ROOT_FOLDER}"
        echo "    --headless"
        echo "    --ignore_existing"
        echo "    --max_radius ${MAX_RADIUS}"
        echo

        ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
            exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
            --file "${ply_file}" \
            --matrices "$(dirname ${ply_file})/matrices_for_rendering.txt" \
            --output_path "${OUTPUT_ROOT_FOLDER}" \
            --headless \
            --ignore_existing \
            --max_radius "${MAX_RADIUS}"
    done
fi

if [[ "$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.type')" == "COLMAP" ]]; then
    PLY_PATH=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.ply_path')
    if [[ -z "${OUTPUT_ROOT}" ]]; then OUTPUT_ROOT="${ROOT_TO_PROCESS}"; fi

    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
    echo "    --file ${PLY_PATH}"
    echo "    --matrices ${ROOT_TO_PROCESS}/test/matrices_for_rendering.json"
    echo "    --output_path ${OUTPUT_ROOT}"
    echo "    --headless"
    echo "    --ignore_existing"
    echo "    --max_radius ${MAX_RADIUS}"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
        --file "${PLY_PATH}" \
        --matrices "${ROOT_TO_PROCESS}/test/matrices_for_rendering.json" \
        --output_path "${OUTPUT_ROOT}" \
        --headless \
        --ignore_existing \
        --max_radius "${MAX_RADIUS}"

    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
    echo "    --file ${PLY_PATH}"
    echo "    --matrices ${ROOT_TO_PROCESS}/train/matrices_for_rendering.json"
    echo "    --output_path ${OUTPUT_ROOT}"
    echo "    --headless"
    echo "    --ignore_existing"
    echo "    --max_radius ${MAX_RADIUS}"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
        --file "${PLY_PATH}" \
        --matrices "${ROOT_TO_PROCESS}/train/matrices_for_rendering.json" \
        --output_path "${OUTPUT_ROOT}" \
        --headless \
        --ignore_existing \
        --max_radius "${MAX_RADIUS}"

    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
    echo "    --file ${PLY_PATH}"
    echo "    --matrices ${ROOT_TO_PROCESS}/val/matrices_for_rendering.json"
    echo "    --output_path ${OUTPUT_ROOT}"
    echo "    --headless"
    echo "    --ignore_existing"
    echo "    --max_radius ${MAX_RADIUS}"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
        --file "${PLY_PATH}" \
        --matrices "${ROOT_TO_PROCESS}/val/matrices_for_rendering.json" \
        --output_path "${OUTPUT_ROOT}" \
        --headless \
        --ignore_existing \
        --max_radius "${MAX_RADIUS}"
fi

if [[ "$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.type')" == "CANDIDATES" ]]; then
    PLY_PATH=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.ply_path')
    if [[ -z "${OUTPUT_ROOT}" ]]; then OUTPUT_ROOT="${ROOT_TO_PROCESS}"; fi

    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
    echo "    --file ${PLY_PATH}"
    echo "    --matrices=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.matrices // "${ROOT_TO_PROCESS}/matrices_for_rendering.json"')"
    echo "    --output_path ${OUTPUT_ROOT}"
    echo "    --headless"
    echo "    --ignore_existing"
    echo "    --max_radius ${MAX_RADIUS}"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
        --file "${PLY_PATH}" \
        --matrices=$(cat params.yaml | ~/.homebrew/bin/yq -r '.cpp_render_'$sub'.matrices // "'${ROOT_TO_PROCESS}'/matrices_for_rendering.json"') \
        --output_path "${OUTPUT_ROOT}" \
        --headless \
        --ignore_existing \
        --max_radius "${MAX_RADIUS}"
fi
