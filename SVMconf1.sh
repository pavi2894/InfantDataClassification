#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=3gb
#SBATCH --constraint avx2

python SVM.py SVMyamls/SVMyaml1.yaml
