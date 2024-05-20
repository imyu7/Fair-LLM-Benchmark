#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=0:30:00
#$ -j y
#$ -N generate_text
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.9 cuda/11.8 cudnn/8.6
conda activate bias

huggingface-cli login --token $HF_TOKEN
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python Equity-Evaluation-Corpus/src/generation.py -m $MODEL