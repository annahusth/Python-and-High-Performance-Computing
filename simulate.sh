#!/bin/sh
#BSUB -q hpc
#BSUB -J jacobi
#BSUB -o jacobioutput/jacobi.out
#BSUB -e jacobioutput/jacobi.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 01:00



source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python simulate.py 10