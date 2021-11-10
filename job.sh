#!/bin/bash
#SBATCH -J =ActionClip=
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

#last-log
#no-pretrain-
#resnet-50-test
if [ -f "/home/10501001/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/10501001/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/10501001/anaconda3/bin:$PATH"
fi
conda activate ACTION-CLIP

module load cudnn7.6-cuda10.2/
module load cuda10.0/
module load nccl2-cuda10.2-gcc/

export MASTER_ADDR = localhost
export MASTER_PORT = 5678

nvidia-smi
#cd ~/projects/ActionCLIP/work
echo 'ActionClip boot~'

#bash scripts/run_train.sh  ./configs/ucf101/ucf_train.yaml

#if [ -f $1 ]; then
#  config=$1
#else
#  echo "need a config file"
#  exit
#fi

#config=configs/ucf101/ucf_train.yaml
config=configs/k400/k400_train.yaml

type=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['type'])")
arch=$(python -c "import yaml;print(yaml.load(open('${config}'))['network']['arch'])")
dataset=$(python -c "import yaml;print(yaml.load(open('${config}'))['data']['dataset'])")
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p exp/${type}/${arch}/${dataset}/${now}
python -u train.py  --config ${config} --log_time $now 2>&1|tee exp/${type}/${arch}/${dataset}/${now}/$now.log
