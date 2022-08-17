python /home/kremeto1/neural_rendering/artwin/transform_data_to_fvs_format.py \
    --src /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29 \
    --out /nfs/projects/artwin/experiments/kremeto1-freeViewSynthesis/training_data/tanks_temples/training/artwin/dense/2019-09-28_08.31.29 \
    --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_08.31.29/2019-09-28_08.31.29.ply \
    --verbose

python /home/kremeto1/neural_rendering/artwin/transform_data_to_fvs_format.py \
    --src /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53 \
    --out /nfs/projects/artwin/experiments/kremeto1-freeViewSynthesis/training_data/tanks_temples/training/artwin/dense/2019-09-28_16.11.53 \
    --ply_path /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53/2019-09-28_16.11.53.ply \
    --verbose

python3 /nfs/projects/artwin/experiments/kremeto1-freeViewSynthesis/data/create_data_central_crop_artwin.py \
    --src /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53-rendered \
    --colmap-src /nfs/projects/artwin/experiments/as_colmap_60_fov_pyrender/2019-09-28_16.11.53 \
    --out /nfs/projects/artwin/experiments/kremeto1-freeViewSynthesis/training_data/ciirc/artwin/dense/2019-09-28_08.31.29 \
    --verbose
