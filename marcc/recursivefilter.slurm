#!/bin/bash
#SBATCH --job-name=recursivefilter
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nwagle1@jhu.edu
#SBATCH --time=1-23:00:00
#SBATCH --account=kgreen66

set -e
module load python/3.7
module load cuda/10.1
source $HOME/code/Aeyemovment/venv/bin/activate

../venv/bin/python3 ./create_filtered_images.py ../data/images/frames ../data/images/v5-0.15s 2 0.15
