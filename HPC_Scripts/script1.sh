#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/ML_bachelor_%J.out
#BSUB -e HPC_outputs/ML_bachelor_%J.err
#BSUB -q gpuv100
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 12:00
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

python src/model1/embed_batches.py
python src/model1/embed_valid_sample.py
python src/model1/predict.py
