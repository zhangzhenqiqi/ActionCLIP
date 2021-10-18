#!/bin/bash
#SBATCH -J =ActionClip=
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

if [ -f "/home/10501001/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/10501001/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/10501001/anaconda3/bin:$PATH"
fi
conda activate ACTION-CLIP

module load cudnn7.6-cuda10.2/
module load cuda10.0/
module load nccl2-cuda10.2-gcc/


cd ~/projects/ActionCLIP/work
echo 'ActionClip boot~'


python test.py
