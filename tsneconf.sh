#!/bin/bash
#SBATCH --time=4-10
#SBATCH --mem-per-cpu=5gb
#SBATCH --gres=gpu:1
#SBATCH --constraint avx2

python CPC_TSNEExperiment.py TSNEyamls/tsneyaml1.yaml
