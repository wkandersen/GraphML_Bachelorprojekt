#!/bin/bash
#BSUB -J Graph_ML_Bachelor_vector
#BSUB -o ML_bachelor_vector_%J.out
#BSUB -e ML_bachelor_vector_%J.err
#BSUB -q gpua10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:01
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python test.py