import argparse
import os
import sys 
sys.path.append("../..") 

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import vision_transformer as vits
from methods.vpt.prompters import PadPrompter, PatchPrompter

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead

from config import clip_pretrain_path, dino_pretrain_path


parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
parser.add_argument('--use_ssb_splits', action='store_true', default=True)

parser.add_argument('--transform', type=str, default='imagenet')
parser.add_argument('--model', type=str, default='dino')
parser.add_argument('--model_path', type=str)

parser.add_argument('--freq_rep_learn', type=int)
parser.add_argument('--prompt_size', type=int)
parser.add_argument('--prompt_type', type=str, default='all')
parser.add_argument('--pretrained_model_path', type=str)


# ----------------------
# INIT
# ----------------------
args = parser.parse_args()
device = torch.device('cuda')
args = get_class_splits(args)


def test(model, test_loader, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=0, eval_funcs=args.eval_funcs, save_name=save_name)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    print(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # Hyper-paramters
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    args.image_size = 224
    args.feat_dim = 768
    args.proj_dim = 256
    args.num_mlp_layers = 3
    args.patch_size = 16
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_ctgs = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # BASE MODEL
    # ----------------------
    backbone = vits.__dict__['vit_base']().to(device)

    if args.prompt_type == 'patch':
        args.prompt_size = 1
        prompter = PatchPrompter(args)

    elif args.prompt_type == 'all':
        args.prompt_size = 30
        prompter1 = PadPrompter(args)
        args.prompt_size = 1
        prompter2 = PatchPrompter(args)
        prompter = nn.Sequential(prompter1, prompter2)

    print(args)

    # ----------------------
    # CLS HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_ctgs, nlayers=args.num_mlp_layers)
    classifier = nn.Sequential(backbone, projector).cuda()
    model = nn.Sequential(prompter, classifier).cuda()

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.cuda()

    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    model.load_state_dict(state_dict)

    # DATASETS
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # ------------------
    # DATALOADERS
    # --------------------
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    
    # ----------------------
    # EVAL
    # ----------------------
    all_acc, old_acc, new_acc = test(model, test_loader_unlabelled, save_name='Train ACC Unlabelled', args=args)
    print('Best Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
