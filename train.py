# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import os
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.KLLoss import KLLoss
from utils.ContrastiveLoss import ContrastiveLoss
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.Text_Prompt import *
from utils.saving import *


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    print('this is add nam & ste proc!')
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                         config['data']['dataset']),
               settings=wandb.Settings(start_method="fork"))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    # 加载模型，这里的加载并不包含预训练参数？
    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
                                       joint=config.network.joint,
                                       is_action=config.network.is_action, use_sis=config.network.use_sis,
                                       use_nam=config.network.use_nam)  # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    #
    # torch.distributed.init_process_group(backend='nccl', world_size=2,
    #                                      rank=0)
    # model_text = model_text.cuda()
    # model_text = torch.nn.parallel.DistributedDataParallel(model_text)
    # print("OK.1")
    # model_image = model_image.cuda()
    # model_image = torch.nn.parallel.DistributedDataParallel(model_image)
    # print("OK.2")
    # fusion_model = fusion_model.cuda()
    # fusion_model = torch.nn.parallel.DistributedDataParallel(fusion_model)
    # print("OK.3")

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    # wandb.watch(model)
    # wandb.watch(fusion_model)

    def testStr(s, d):
        s = s.lower()
        for c in d:
            if s.find(c) != -1:
                return False
        return True

    # d = ['sigmoid', 'pool', 'dropout', 'relu', 'identity', 'gelu']
    # for name, mod in model.named_modules():
    #     # print('mod type',type(mod))
    #     # print(str(mod))
    #     if len(list(mod.children())) == 0 and testStr(name, d) and testStr(str(mod), d):
    #         print(name, ' : ', mod)
    #         print('-', mod.weight.dtype)

    print('model=' * 20)
    print(model)
    print('fusion_model=' * 20)
    print(fusion_model)

    train_data = Action_DATASETS(config.data.train_list, config.data.label_list, num_segments=config.data.num_segments,
                                 image_tmpl=config.data.image_tmpl, random_shift=config.random_shift,
                                 transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)
    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, random_shift=config.random_shift,
                               num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                               transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=False, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = text_prompt(train_data)

    optimizer = _optimizer(config, model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    best_prec5 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)
        return

    loss_img = KLLoss()
    loss_txt = KLLoss()

    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk, (images, list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
            b, t, c, h, w = images.size()  # 32，8，3，224，224

            # print('=====input image size: ', b, t, c, h, w)

            text_id = numpy.random.randint(num_text_aug, size=len(list_id))
            texts = torch.stack([text_dict[j][i, :] for i, j in zip(list_id, text_id)])

            images = images.to(device).view(-1, c, h,
                                            w)  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            # adapt for tsn-action module
            # images = images.to(device)

            texts = texts.to(device)

            image_embedding = model_image(images)  # (16*8,512)
            image_embedding = image_embedding.view(b, t, -1)

            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)

            if epoch == 0 and kkk == 0:
                print('image_embedding size: ', image_embedding.size())  # (16,512) b=16   (32,1024)b=32
                print('text_embedding size: ', text_embedding.size())  # (16,512) b=16

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()
            # ##origin loss start
            logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding,
                                                              logit_scale)  # 16*16 b=16

            ground_truth = torch.tensor(gen_label(list_id), dtype=image_embedding.dtype, device=device)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)
            total_loss = (loss_imgs + loss_texts) / 2
            # ##end

            # ground_truth = torch.tensor(gen_label(list_id), dtype=image_embedding.dtype, device=device)
            # loss_imgs = loss_img(image_embedding, text_embedding, ground_truth)
            # # loss_texts = loss_txt(text_embedding,image_embedding,ground_truth)
            # total_loss = loss_imgs
            #
            wandb.log({"train_total_loss": total_loss})
            wandb.log({"train_loss_imgs": loss_imgs})
            wandb.log({"train_loss_texts": loss_texts})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        if epoch % config.logging.eval_freq == 0:  # and epoch>0
            prec1, prec5 = validate(epoch, val_loader, classes, device, model, fusion_model, config, num_text_aug)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)

        print('Testing: prec1/best_prec1: {}/{}'.format(prec1, best_prec1))
        print('prec5/best_prec5: {}/{}'.format(prec5, best_prec5))
        print('Saving:')
        filename = "{}/last_model.pt".format(working_dir)

        # epoch_saving(epoch, model, fusion_model, optimizer, filename)
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, optimizer)


if __name__ == '__main__':
    main()
