#!/bin/bash
#SBATCH --job-name=compute
#SBATCH --output=logs/compute_radii_%j.log
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
# export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}
export PATH=/usr/local/bin:/usr/bin

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo

WORKSPACE=/home/kremeto1/neural_rendering

for ply_file in `find /home/kremeto1/neural_rendering/datasets/processed/inloc/inloc_rendered_splatting -type f -name '???_scan_???.ptx.ply'`
do
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/renderer-app.sif ~/containers/radii-compute-app.executable"
    echo "    --file ${ply_file}"
    echo "    -m $(dirname ${ply_file})/matrices_for_rendering.txt"
    echo "    -o $(dirname $(dirname ${ply_file}))"
    echo "    -d"

    if [[ ! -f "$(dirname ${ply_file})_RADII.LOCK" ]]; then
        touch "$(dirname ${ply_file})_RADII.LOCK"
    else
        echo "Already locked $(dirname ${ply_file})_RADII.LOCK, skipping."
        continue
    fi

    if [[ ! -f "${ply_file}.kdtree.radii" ]]; then
        ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
            exec --nv --bind /nfs:/nfs ~/containers/renderer-app.sif ~/containers/radii-compute-app.executable \
            --file "${ply_file}" \
            -m "$(dirname ${ply_file})/matrices_for_rendering.txt" \
            -o "$(dirname $(dirname ${ply_file}))" \
            -d
    else
        echo "Already existing ${ply_file}.kdtree.radii, skipping."
    fi
done
