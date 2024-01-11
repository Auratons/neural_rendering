#!/bin/bash
#SBATCH --job-name=render_candidates
#SBATCH --output=logs/render_candidates_%j.log
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=2
#SBATCH --exclude='amd-[01-02],node-[12,14,15,16,17]'

. /opt/ohpc/admin/lmod/lmod/init/bash
ml purge
module load CUDA/9.1.85
module load cuDNN/7.0.5-CUDA-9.1.85
module load Mesa/18.1.1-fosscuda-2018b

sub=$1

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo

WORKSPACE=/home/kremeto1/neural_rendering

echo
echo "Running:"
echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/inloc/render_candidates.py"
echo "    --src_output=$(cat params.yaml | yq -r '.render_candidates_'$sub'.src_output')"
echo "    --input_poses=$(cat params.yaml | yq -r '.render_candidates_'$sub'.input_poses')"
echo "    --just_jsons"
echo

~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' python $WORKSPACE/inloc/render_candidates.py \
    --src_output=$(cat params.yaml | yq -r '.render_candidates_'$sub'.src_output') \
    --input_poses=$(cat params.yaml | yq -r '.render_candidates_'$sub'.input_poses') \
    --just_jsons

EXECUTABLE=$(cat params.yaml | ~/.homebrew/bin/yq -r '.render_candidates_'$sub'.renderer_executable')
MAX_RADIUS=$(cat params.yaml | ~/.homebrew/bin/yq -r '.render_candidates_'$sub'.max_radius')
OUTPUT_ROOT=$(cat params.yaml | ~/.homebrew/bin/yq -r '.render_candidates_'$sub'.output_root // ""')
ROOT_TO_PROCESS=$(cat params.yaml | yq -r '.render_candidates_'$sub'.src_output')
cd ${ROOT_TO_PROCESS}

for i in `ls *.txt | tac`
do
    PLY_PATH=$(cat $i)
    if [[ -z "${OUTPUT_ROOT}" ]]; then OUTPUT_ROOT="${ROOT_TO_PROCESS}"; fi

    echo
    echo "~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity"
    echo "    exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif ${EXECUTABLE}"
    echo "    --file=${PLY_PATH}"
    echo "    --matrices=${ROOT_TO_PROCESS}/$(echo $i | sed 's/txt/json/')"
    echo "    --output_path=${OUTPUT_ROOT}"
    echo "    --headless"
    echo "    --max_radius=${MAX_RADIUS}"
    echo

    ~/.homebrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' singularity \
        exec --nv --bind /nfs:/nfs ~/containers/splatter-app.sif "${EXECUTABLE}" \
        --file="${PLY_PATH}" \
        --matrices=${ROOT_TO_PROCESS}/$(echo $i | sed 's/txt/json/') \
        --output_path="${OUTPUT_ROOT}" \
        --headless \
        --max_radius="${MAX_RADIUS}"
done
