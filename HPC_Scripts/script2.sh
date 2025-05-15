#!/bin/bash
#BSUB -J Graph_ML_Bachelor_NN
#BSUB -o HPC_outputs/ML_bachelor_NN_%J.out
#BSUB -e HPC_outputs/ML_bachelor_NN_%J.err
#BSUB -q gpuv100
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -W 22:00
#BSUB -n 4
#BSUB -B
#BSUB -N
#BSUB -u williamkirkandersen@gmail.com

nvidia-smi
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/Packages:$(pwd)/dataset$(pwd)/dataset

source ~/miniconda3/bin/activate

conda activate Bachelorprojekt

# python src/ny_model_2/train.py
# python src/train_vector128.py

# kernprof -l -v src/model1/embed_batches_2.py 
python src/model1/embed_batches_2.py --alpha=0.1 --epochs=5 --batch_size=64 --embedding_dim=2 --weight=0.25 > joboutput_${LSB_JOBID}_$(date +%Y%m%d_%H%M%S).out 2>&1
