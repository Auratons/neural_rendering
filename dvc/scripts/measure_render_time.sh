#!/bin/bash
#SBATCH --job-name=measure
#SBATCH --output=logs/render_times_%j.log
#SBATCH --mem=40G
#SBATCH --time=0-12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --exclude='dgx-[5],amd-[01-02],node-[12]'

# jq may not be installed globally, add brew as another option
# Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "Interactive Slurm mode GPU index: ${SLURM_STEP_GPUS}"
echo "Batch Slurm mode GPU index: ${SLURM_JOB_GPUS}"
echo "PID: ${BASHPID}"
echo

WORKSPACE=/home/kremeto1/neural_rendering

for pipeline in hagia_sophia_interior grand_place_brussels; do
    for setup in 4 5 6 10 11 12; do
        for pixels in 512 256; do
            params=${WORKSPACE}/dvc/pipeline-${pipeline}/params.yaml
            ply=$(cat ${params} | yq -r '.render_colmap_'${setup}'.ply_file')

            temp=$(basename $(cat ${params} | yq -r '.render_colmap_'${setup}'.src_output'))
            part=${temp#${pipeline}_minsz-512_valr-0.2_pts-2.0_}

            # --nodelist=node-[04,05,07-10,12]
            # srun --job-name=measure \
            #     --output=${pipeline}_${part}_sz-${pixels}_%j.log \
            #     --mem=16G \
            #     --time=00:10:00 \
            #     --partition=compute \
            #     --cpus-per-task=16 \
            mkdir -p ${WORKSPACE}/colmap/render_times
            echo ~/.homebrew/bin/time -f='real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' \
                    $(which python) ${WORKSPACE}/colmap/measure_render_time.py \
                        --colmap_sparse_dir=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/sparse \
                        --ply_path=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/${ply} \
                        --render_size=${pixels} \
                        --voxel_size=$(cat ${params} | yq -r '.render_colmap_'${setup}'.voxel_size') \
                        --count=200 | tee ${WORKSPACE}/colmap/render_times/${pipeline}_${part}_sz-${pixels}.log
            # export SLURM_CPUS_ON_NODE=16
            # cpulimit \
            #     --path=/home/kremeto1/.homebrew/bin/time \
            #     --monitor-forks \
            #     --cpu=1600 \
            #     --verbose \
                ~/.homebrew/bin/time -f='real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' \
                    python ${WORKSPACE}/colmap/measure_render_time.py \
                        --colmap_sparse_dir=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/sparse \
                        --ply_path=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/${ply} \
                        --render_size=${pixels} \
                        --voxel_size=$(cat ${params} | yq -r '.render_colmap_'${setup}'.voxel_size') \
                        --count=200 | tee ${WORKSPACE}/colmap/render_times/${pipeline}_${part}_sz-${pixels}.log
        done
    done
done
