#!/bin/sh
#BSUB -q c02613
#BSUB -J jacobi_cuda
#BSUB -o jacobi_cuda_output/jacobi_cuda.out
#BSUB -e jacobi_cuda_output/jacobi_cuda.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 01:00
#BSUB -gpu "num=1:mode=exclusive_process"



source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python CUDA.py 10