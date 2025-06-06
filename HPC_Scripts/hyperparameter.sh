#!/bin/bash
#BSUB -J Graph_ML_Bachelor
#BSUB -o HPC_outputs/sweep_lambda_alpha_%J.out
#BSUB -e HPC_outputs/sweep_lambda_alpha_%J.err
#BSUB -q c02613
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 00:29
#BSUB -B
#BSUB -N
#BSUB -u s224197@dtu.dk

export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset

source ~/miniconda3/bin/activate
conda activate Bachelorprojekt

export WANDB_API_KEY="b26660ac7ccf436b5e62d823051917f4512f987a"

python src/model1/wandb_hyper_sweep_2.py
