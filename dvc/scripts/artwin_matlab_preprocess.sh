#!/bin/bash
#SBATCH --job-name=artwin
#SBATCH --output=artwin_%j.log
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=20

ml purge
ml load MATLAB/2019b

# jq may not be installed globally, add brew as another option
# # Also, conda is not activateing the environment
export PATH=~/.conda/envs/pipeline/bin:~/.linuxbrew/bin:${PATH}

echo
echo "Running on $(hostname)"
echo "The $(type python)"
echo

WORKSPACE=/home/kremeto1/neural_rendering

cd $WORKSPACE/artwin
~/.linuxbrew/bin/time -f 'real\t%e s\nuser\t%U s\nsys\t%S s\nmemmax\t%M kB' matlab -batch $1
