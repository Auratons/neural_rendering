#!/bin/bash

# Conda is not activating the environment,
# cpulimit is installed by homebrew.
# export PATH=~/.conda/envs/test/bin:~/.homebrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo "PID: ${BASHPID}"
echo

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKSPACE="$(realpath "${SCRIPT_DIR}/../../")"
for pipeline in grand_place_brussels hagia_sophia_interior pantheon_exterior; do
    for setup in $(seq 12); do
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
            echo cpulimit \
                --path=/home/kremeto1/.homebrew/bin/time \
                --monitor-forks \
                --cpu=1600 \
                --  -f='real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' \
                    $(which python) ${WORKSPACE}/colmap/measure_render_time.py \
                        --colmap_sparse_dir=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/sparse \
                        --ply_path=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/${ply} \
                        --render_size=${pixels} \
                        --voxel_size=$(cat ${params} | yq -r '.render_colmap_'${setup}'.voxel_size') \
                        --count=400 | tee ${WORKSPACE}/colmap/render_times/${pipeline}_${part}_sz-${pixels}.log
            export SLURM_CPUS_ON_NODE=16
            cpulimit \
                --path=/home/kremeto1/.homebrew/bin/time \
                --monitor-forks \
                --cpu=1600 \
                --verbose \
                --  -f='real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' \
                    $(which python) ${WORKSPACE}/colmap/measure_render_time.py \
                        --colmap_sparse_dir=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/sparse \
                        --ply_path=$(cat ${params} | yq -r '.render_colmap_'${setup}'.root')/${ply} \
                        --render_size=${pixels} \
                        --voxel_size=$(cat ${params} | yq -r '.render_colmap_'${setup}'.voxel_size') \
                        --count=400 | tee ${WORKSPACE}/colmap/render_times/${pipeline}_${part}_sz-${pixels}.log
            exit 0
        done
    done
done
