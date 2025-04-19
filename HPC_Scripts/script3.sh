#!/bin/bash
#BSUB -J Graph_ML_Bachelor_vector
#BSUB -o HPC_outputs/ML_bachelor_boost_%J.out
#BSUB -e HPC_outputs/ML_bachelor_boost_%J.err
#BSUB -q gpua10
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 3:00
#BSUB -n 4
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/xgboost_model.py