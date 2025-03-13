import os
import torch
import numpy as np
import argparse
import random

from dassl.config import get_cfg_default
from yacs.config import CfgNode as CN

from train_utils import CUSTOM_TEMPLATES
from clip import clip

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers

from forget_cls import *

from train_forget import load_test_datasets, initialize_config
from clip_forget_imagegen import generate_images
from clip_forget_utils import clip_classifier

torch.set_num_threads(5)

parser = argparse.ArgumentParser()
parser.add_argument("--high_prob", default=0.8, type=float)
parser.add_argument("--run_ds", type=str, default="StanfordDogs,StanfordCars,Caltech101,OxfordFlowers")
parser.add_argument("--forget_labels", type=str, default="")
parser.add_argument("--backbone_arch", type=str, default="")
parser.add_argument("--sub_prob", type=float, default=0.025)
parser.add_argument("--seed", type=int, default=0, help="only positive value enables a fixed seed")


args = parser.parse_args()
args.config_file = "configs/trainers/adam_lr2e-4_B256_ep200_ViT16.yaml"
print("Arguments : ", args)

device = 'cuda:0'
MODEL_ARCH = args.backbone_arch
if MODEL_ARCH == "ViT-B/16":
    MODEL_ARCH = "ViTB16"

SAVE_PATH = f"generated_{MODEL_ARCH}/"
    

print("SAVE_PATH", SAVE_PATH)

args.run_ds = [item for item in args.run_ds.split(',')]

cfg = initialize_config(args)
test_datasets, test_dataloaders, datasets_cls, tfm_train, tfm_test = load_test_datasets(cfg)


myseed=0
torch.manual_seed(myseed)
random.seed(myseed)
np.random.seed(myseed)
backbone_name = args.backbone_arch
url = clip._MODELS[backbone_name]
model_path = clip._download(url)

print("Loading model..")
try:
    # loading JIT archive
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = None

except RuntimeError:
    state_dict = torch.load(model_path, map_location="cpu")
    
model = clip.build_model(state_dict or model.state_dict()).float().to(device).eval()

kwargs = {'high_prob' : args.high_prob, 'sub_prob' : args.sub_prob}


for main_ds in args.run_ds:
    print(f"main ds: {main_ds}")
    
    main_dataset = test_datasets[main_ds]
    
    if args.forget_labels:
        args.forget_labels = [item for item in args.forget_labels.split(',')]
    else:
        args.forget_labels = forget_classes_all[main_ds]

    # pick forget labels
    forget_labels = forget_classes_all[main_ds]
    clip_weights = clip_classifier(main_dataset.classnames, [CUSTOM_TEMPLATES[main_ds]], model).to(device)
        
    print(f"forget_labels {args.forget_labels}")
    for forget_label in args.forget_labels:
        cls_names = [forget_label]

        id_lbl = main_dataset.classnames.index(forget_label)
        generate_images(model,clip_weights, main_ds, id_lbl, forget_label, tfm_test, device, 
                        save_path= SAVE_PATH, verbose=True, **kwargs)
        
    args.forget_labels = ""