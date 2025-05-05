#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q gpuv100
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -n 4
#BSUB -W 0:15
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com


export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset$(pwd)/dataset

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/Embed_dataset.py