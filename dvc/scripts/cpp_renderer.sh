#!/usr/bin/env bash
#SBATCH --job-name=compute
#SBATCH --output=logs/cpp_render_%j.log
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='dgx-[2,3,4,5],amd-01,node-[12,14,15,16,17]'

nvidia-smi

export PATH=/usr/local/bin:/usr/bin

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo

for ply_file in `find /home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting -type f -name '???_scan_???_30M.ptx.ply'`
do
    if [[ ! -f "$(dirname ${ply_file})_SPLAT.LOCK" ]]; then
        touch "$(dirname ${ply_file})_SPLAT.LOCK"
    else
        echo "Already locked $(dirname ${ply_file})_SPLAT.LOCK, skipping."
        continue
    fi

    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs --env 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib' ~/containers/splatter-app.sif ~/containers/splatter-app.executable"
    echo "    --file ${ply_file}"
    echo "    --matrices $(dirname ${ply_file})/matrices_for_rendering.txt"
    echo "    --output_path $(dirname $(dirname ${ply_file}))"
    echo "    --headless"
    echo "    --ignore_existing"

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs --env 'LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib' ~/containers/splatter-app.sif ~/containers/splatter-app.executable \
        --file "${ply_file}" \
        --matrices "$(dirname ${ply_file})/matrices_for_rendering.txt" \
        --output_path "$(dirname $(dirname ${ply_file}))" \
        --headless \
        --ignore_existing
done
