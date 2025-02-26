import torch
from torch.utils.data import DataLoader, Dataset

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers

from dassl.data.data_manager import build_data_loader
from dassl.data.datasets.build import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.config import get_cfg_default

import numpy as np
import random
import pickle
import os

from typing import Tuple

from forget_cls import *
from clip_forget_utils import *

CUSTOM_TEMPLATES = {
                "OxfordFlowers": "a photo of a {}, a type of flower.",
                "StanfordCars": "a photo of a {}.",
                "Caltech101": "a photo of a {}.",
                "StanfordDogs": "a photo of a {}.",
                }


import clip_forget_utils
clip_forget_utils.CUSTOM_TEMPLATES = CUSTOM_TEMPLATES

device = 'cuda'

def set_seeds(seed: int) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): The seed value to set for random number generators.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_config(args: argparse.Namespace) -> object:
    """
    Initialize and configure the default configuration.

    Args:
        args (argparse.Namespace): Command line arguments.

    """
    cfg = get_cfg_default()
    cfg.merge_from_file(args.config_file)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.SEED = args.seed
    cfg.DATASET.ROOT = "/app/datasets/"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATASET.NUM_SHOTS = -1
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 4
    cfg.DATALOADER.TEST.BATCH_SIZE = 128
    return cfg


def load_test_datasets(cfg: object) -> Tuple[dict, dict, dict, object, object]:
    """
    Load test datasets and create corresponding dataloaders.

    Args:
        cfg (object): Configuration object with dataset settings.

    """
    test_datasets, test_dataloaders, datasets_cls = {}, {}, {}
    for ds in all_ds:
        cfg.DATASET.NAME = ds
        tfm_train = build_transform(cfg, is_train=True)
        tfm_test = build_transform(cfg, is_train=False)
        dataset = build_dataset(cfg)

        test_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.test,
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=None
        )

        test_datasets[ds] = dataset
        test_dataloaders[ds] = test_loader
        datasets_cls[ds] = dataset

    return test_datasets, test_dataloaders, datasets_cls, tfm_train, tfm_test

def load_generated_data(model_arch: str) -> dict:
    """
    Load generated data.

    Args:
        model_arch (str): Model architecture identifier ("RN50" or "ViT-B/16").

    """
    dataloader_generated = {}
    for ds in all_ds:
        # load_dir = f"generated_high_{model_arch}/{ds}/"
        load_dir = f"generated_{model_arch}/{ds}/"
        dataloader_generated[ds] = {}
        
        for lbl in forget_classes_all.get(ds, []):
            lbl_file = f"{load_dir}{lbl}.pt"
            if os.path.exists(lbl_file):
                dataloader_generated[ds][lbl] = torch.load(lbl_file, map_location='cpu')
                print(f"Loaded {lbl} examples for {ds} from {load_dir}")
            else:
                print(f"Skipping {lbl} for {ds}")
    
    return dataloader_generated

def load_results(args: argparse.Namespace) -> dict:
    """
    Load zero-shot results from CLIP model.

    Args:
        args (argparse.Namespace): Command line arguments.

    """
    filename = "results_zs_all_RN50.pkl" if args.backbone_arch == "RN50" else "results_zs_all_ViT16.pkl"
    with open(f"generated/{filename}", "rb") as f:
        return pickle.load(f)
    
def set_param_schedules(init_topvis: int, init_toptext: int, backbone: str) -> Tuple[list, list, list]:
    """
    Set parameter schedules for updating model layers during forgetting.

    Args:
        init_topvis (int): Initial number of visual encoder layers to update.
        init_toptext (int): Initial number of textual encoder layers to update.
        backbone (str): Model architecture ("RN50" or "ViT-B/16").

    """
    std_scheduler = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2]
    if backbone == "RN50":
        text_schedule = [init_toptext + i for i in [0, 2, 7, 13, 20, 25, 30, 35, 40]]
        img_schedule = [init_topvis + i for i in [0, 2, 5, 8, 10, 15, 20]]
    else:
        text_schedule = [init_toptext + i for i in [0, 2, 5, 8, 10, 15, 20]]
        img_schedule = [init_topvis + i for i in [0, 2, 5, 8, 10, 15, 20]]
    return text_schedule, img_schedule, std_scheduler


def set_hyperparameters(args: argparse.Namespace) -> Tuple[int, int]:
    """
    Set initial hyperparameters for `set_param_schedules` function
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    if args.backbone_arch == "RN50":
        return 5, 5
    return 25, 25

    
def evaluate_and_log(args: argparse.Namespace, model: torch.nn.Module, datasets_cls: torch.utils.data.Dataset, 
                     main_ds: str, forget_label: str, test_dataloaders: dict, 
                     train_loader_eval: torch.utils.data.DataLoader, 
                     valid_loader_eval_true: torch.utils.data.DataLoader or None,
                     all_logs: dict, eval_forgetonly: bool = True, debug: bool = False, 
                     classnames: list or None = None) -> Tuple[dict, float]:
    """
    Evaluates the model on various datasets and logs performance metrics.
    
    Args:
        args (argparse.Namespace): Command line arguments.
        model (torch.nn.Module): The CLIP model to evaluate.
        datasets_cls (torch.utils.data.Dataset): Dictionary of dataset classes.
        main_ds (str): Main dataset name.
        forget_label (str): Label to forget.
        test_dataloaders (dict): Test dataloaders.
        train_loader_eval (DataLoader): Loader for training evaluation data.
        valid_loader_eval_true (DataLoader or None): Loader for true validation data.
        all_logs (dict): Dictionary to store logs.
        eval_forgetonly (bool): Whether to evaluate only the forget label.
        debug (bool): Debug mode.
        classnames (list or None): List of class names for the dataset.

    """
    
    if not args.generated_data:
        results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders, train_loader_eval, 
                             eval_forgetonly= eval_forgetonly, debug=debug, device=device)
        acc = results_ds[main_ds][forget_label]['acc_train']

    else:
        clip_weights = clip_classifier(classnames, [CUSTOM_TEMPLATES[main_ds]], model).to(device)
        results_ds = eval_all_ds(model, datasets_cls, main_ds, forget_label, test_dataloaders, 
                             train_loader=None, eval_forgetonly= eval_forgetonly, debug=debug, device=device)
        acc = evaluate_clip_zs(model, train_loader_eval, clip_weights, device=device, out_conf=False)
        acc_val_true_data = evaluate_clip_zs(model, valid_loader_eval_true, clip_weights, device=device, out_conf=False)
        results_ds[main_ds][forget_label]['acc_val_hard'] = acc
        results_ds[main_ds][forget_label]['acc_val_true_data'] = acc_val_true_data
    return results_ds, acc