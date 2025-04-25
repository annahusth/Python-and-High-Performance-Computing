#!/bin/sh
#BSUB -q c02613
#BSUB -J jacobi_cupy
#BSUB -o jacobi_cupy_output/jacobi_cupy.out
#BSUB -e jacobi_cupy_output/jacobi_cupy.err
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -W 01:00
#BSUB -gpu "num=1:mode=exclusive_process"



source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python CuPy.py 10