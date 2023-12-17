# Neural Rerendering in the Wild

This repository is a fork of https://github.com/google/neural_rerendering_in_the_wild
extended with codes for experiments from https://github.com/Auratons/master_thesis,
targeting multiple datasets.


## Repository structure

Top-level files are mostly taken from the upstream repository. The `dvc` folder is
the main entrypoint to experiments. (The folder name is a remnant of a trial to use
[DVC](https://dvc.org) which proven itself to be an unsuitable tool for datasets
comprised of many small files.) The `datasets` folder should contain most of the data and
their transformations. These are referenced from within the `dvc` folder. The `models`
folder is target of trained networks. Finally, `artwin`, `colmap` and `inloc` folders
contain transformation scripts for three datasets families used in the thesis, ARTwin
Dataset, [Image Matching Challenge data](https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/),
and [InLoc Dataset](http://www.ok.sc.e.titech.ac.jp/INLOC/), respectively.


## Dependencies & Runtime

The runtime for the project was Slurm-based compute cluster with graphical capabilities
operated by [Czech Institute of Informatics, Robotics and Cybernetics](https://cluster.ciirc.cvut.cz).
Thus, in `dvc/scripts` folder, there are mentions of SBATCH directives meant as Slurm
scheduler limits and compute requirements for various workloads. In the folder, there
are mentioned also other projects: [InLoc](https://github.com/Auratons/inloc) with codes
for InLoc dataset transformations as well as InLoc algorithm,
[Splatter Renderer](https://github.com/Auratons/renderer_surface_splatting), and
[Ray Marcher Renderer](https://github.com/Auratons/renderer_ray_marching).

In the scripts are mentioned also binaries `time` as `gnu-time`,
[`cpulimit`](https://sourceforge.net/projects/limitcpu/files/limitcpu/) and
Python's `yq` (accepts -r option from underlying jq).


## Data

Raw data should be stored in `datasets/raw/inloc` and
`datasets/raw/imc/<grand_place_brussels|hagia_sophia_interior|pantheon_exterior>`,
the ARTwin dataset is not open to public, so it resided elsewhere on the cluster
storage. Other dependencies of the rerendering model are (from the upstream project's
README):
*   Download the DeepLab semantic segmentation model trained on the ADE20K
    dataset from this link:
    http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz
*   Unzip the downloaded file to: $base_dir/deeplabv3_xception_ade20k_train
*   Download this [file](https://github.com/MoustafaMeshry/vgg_loss/blob/master/vgg16.py) for an implementation of a vgg-based perceptual loss.
*   Download trained weights for the vgg network as instructed in this link: https://github.com/machrisaa/tensorflow-vgg
*   Save the vgg weights to $base_dir/vgg16_weights/vgg16.npy


## Running codes

Prepare conda environment from `environment.yml`, go to specific subfolder of `dvc/pipeline-*`
depending on which dataset should be targeted and run `sbatch ../scripts/<SCRIPT_NAME> <CONFIG_NAME>`.
Scripts always read `params.yaml` file and pick proper configuration key `<PREFIX>_<CONFIG_NAME>`,
where `<PREFIX>` is first part of each top-level YAML key in the parameters file and it varies
across scripts. To find out what the prefix is for given script file, please refer to that script
contents.
