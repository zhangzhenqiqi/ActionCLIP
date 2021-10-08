# This is an official pytorch implementation of ActionCLIP: A New Paradigm for Video Action Recognition [[arXiv]](https://arxiv.org/abs/2109.08472)
[Fork from here](https://github.com/sallymmx/actionclip#data-preparation)
## Overview

![ActionCLIP](ActionCLIP.png)

## Content 
 - [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Updates](#updates)
- [Pretrained Models](#pretrained-models)
  * [Kinetics-400](#kinetics-400)
  * [Hmdb51 && UCF101](#HMDB51&&UCF101)
- [Testing](#testing)
- [Training](#training)
- [Contributors](#Contributors)
- [Citing_ActionClip](#Citing_ActionCLIP)
- [Acknowledgments](#Acknowledgments)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) >= 1.8
- [wandb](https://wandb.ai/)
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

More detail information about libraries see [INSTALL.md](INSTALL.md).

## Data Preparation
We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.
We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/),
[Charades](https://prior.allenai.org/projects/charades). 
 
## Updates
- We now support single crop validation(including zero-shot) on Kinetics-400, UCF101 and HMDB51. The pretrained models see [MODEL_ZOO.md](MODEL_ZOO.md) for more information.
- we now support the model-training on Kinetics-400, UCF101 and HMDB51 on 8, 16 and 32 frames. The model-training configs see [configs/README.md](configs/README.md) for more information.
- We now support the model-training on your own datasets. The detail information see  [configs/README.md](configs/README.md).

## Pretrained Models
Training video models is computationally expensive. Here we provide some of the pretrained models.
We provide a large set of trained models in the ActionCLIP [MODEL_ZOO.md](MODEL_ZOO.md).

### Kinetics-400
We experiment ActionCLIP with different backbones(we choose Transf as our final visual
prompt since it obtains the best results) and input frames configurations on k400. Here is a list of pre-trained models that we provide (see Table 6 of the paper).

 | model             | n-frame     | top1 Acc(single-crop) | top5 Acc(single-crop)| checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:-------------: |:---------------------------------------------------------: | 
|ViT-B/32 | 8 | 78.36%          | 94.25%|[link](https://pan.baidu.com/s/1Ok8HG4lb3kePKbtBExJo2g) pwd:8hg2 
| ViT-B/16  | 8 |   81.09%    | 95.49% |[link]() 
| ViT-B/16 | 16 | 81.68%  | 95.87% |[link]() 
| ViT-B/16 | 32 |82.32%    | 96.20% |[link](https://pan.baidu.com/s/1t3wROD0rLHQkxB2yD7TTkA) pwd:v7nn                                                       

### HMDB51 && UCF101
On HMDB51 and UCF101 datasets, the accuracy(k400 pretrained) is reported under the accurate setting.

#### HMDB51
| model             | n-frame     | top1 Acc(single-crop) | checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:---------------------------------------------------------: | 
|ViT-B/16 | 32 | 76.2%          | [link]() 

#### UCF101
| model             | n-frame     | top1 Acc(single-crop) | checkpoint                                                   |
| :-----------------: | :-----------: | :-------------: |:---------------------------------------------------------: | 
|ViT-B/16 | 32 | 97.1%          | [link]() 

## Testing 
To test the downloaded pretrained models on Kinetics or HMDB51 or UCF101, you can run `scripts/run_test.sh`. For example:
```
# test
bash scripts/run_test.sh  ./configs/k400/k400_ft_tem.yaml

```
### Zero-shot
We provide several examples to do zero-shot validation on kinetics-400, UCF101 and HMDB51.
- To do zero-shot validation on Kinetics from CLIP pretrained models, you can run:
```
# zero-shot
bash scripts/run_test.sh  ./configs/k400/k400_ft_zero_shot.yaml
```
- To do zero-shot validation on UCF101 and HMDB51 from Kinetics pretrained models, you need first prepare the k400 pretrained model and then you can run:
```
# zero-shot
bash scripts/run_test.sh  ./configs/hmdb51/hmdb_ft_zero_shot.yaml

```


## Training
We provided several examples to train ActionCLIP  with this repo:
- To train on Kinetics from CLIP pretrained models, you can run:
```
# train 
bash scripts/run_train.sh  ./configs/k400/k400_ft_tem_test.yaml
```
- To train on HMDB51 from Kinetics400 pretrained models, you can run:
```
# train 
bash scripts/run_train.sh  ./configs/hmdb51/hmdb_ft.yaml
```
- To train on UCF101 from Kinetics400 pretrained models, you can run:
```
# train 
bash scripts/run_train.sh  ./configs/ucf101/ucf_ft.yaml
```
More training details, you can find in  [configs/README.md](configs/README.md)

## Contributors
ActionCLIP is written and maintained by [Mengmeng Wang](https://sallymmx.github.io/) and [Jiazheng Xing](https://april.zju.edu.cn/team/jiazheng-xing/).

## Citing ActionCLIP
If you find ActionClip useful in your research, please cite our paper.
```
@misc{wang2021actionclip,
      title={ActionCLIP: A New Paradigm for Video Action Recognition}, 
      author={Mengmeng Wang and Jiazheng Xing and Yong Liu},
      year={2021},
      eprint={2109.08472},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
# Acknowledgments
Our code is based on [CLIP](https://github.com/openai/CLIP) and [STM](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jiang_STM_SpatioTemporal_and_Motion_Encoding_for_Action_Recognition_ICCV_2019_paper.pdf).
