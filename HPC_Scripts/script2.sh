#!/bin/bash
#BSUB -J Graph_ML_Bachelor_NN
#BSUB -o HPC_outputs/ML_bachelor_NN_%J.out
#BSUB -e HPC_outputs/ML_bachelor_NN_%J.err
#BSUB -q gpua40
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
#BSUB -n 4
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com


export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset$(pwd)/dataset

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

# python src/ny_model_2/train.py
python src/train_vector128.py