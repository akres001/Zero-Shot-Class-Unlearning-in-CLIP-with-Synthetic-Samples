import numpy as np
import torch
from clip import clip

from PIL import Image
from tqdm import tqdm
import random

from torchvision import transforms
import torchvision.transforms as T
from clip_forget_imagegen import rescale
from sklearn.metrics import confusion_matrix

import json
import pandas as pd
from clip_forget_losses import *

import datasets.stanford_cars
import datasets.stanford_dogs
import datasets.caltech101
import datasets.oxford_flowers

import pickle
import argparse

from typing import Tuple

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CUSTOM_TEMPLATES = {}


def train_epoch(batch: Tuple[torch.Tensor, torch.Tensor], model: torch.nn.Module, 
                transform_gaussian: object, n_samples: int, device: str = 'cpu') -> torch.Tensor:
    """
    Train the model for one epoch on a single batch.

    Args:
        batch (tuple[torch.Tensor, torch.Tensor]): Batch containing images and text tokens.
        model (torch.nn.Module): The CLIP model to train.
        transform_gaussian (object): Gaussian noise transform.
        n_samples (int): Number of samples for loss computation.
        device (str): Device to perform computation on ('cpu' or 'cuda').
    """
    model.train()
    freeze_batch_norm(model)

    image, texts = batch
    image, texts = image.to(device), texts.to(device)
    
    loss = lipschitz_loss_joint(model, texts, image, transform_gaussian, 
                                n_samples=n_samples, device=device, verbose=True)
    return loss

def ds_relevant_forget(dataset: torch.utils.data.Dataset, label_name: str, data_type: str = 'train') -> list:
    """
    Extract relevant data instances for a specific label from a dataset.

    Args:
        dataset (Dataset): Dataset object with classnames and data splits.
        label_name (str): Name of the label to filter by.
        data_type (str): Type of data split ('train', 'valid', or 'test').

    """
    
    cls_names = [label_name]
    
    id_lbl = dataset.classnames.index(label_name)

    relevant_trainx = []
    if data_type=='train':
        data_loop = dataset.train_x 
    elif data_type=='valid':
        data_loop = dataset.val 
    elif data_type=='test':
        data_loop = dataset.test     
    
    
    for el in data_loop:
        if dataset.lab2cname[el.label] in [label_name]:
            relevant_trainx.append(el)

    return relevant_trainx


def get_important_params_mask(model: torch.nn.Module, topk_v: int = 5, topk_t: int = 5, 
                              freeze: set or None = None) -> dict:
    """
    Determine which model parameters are important based on gradient magnitude.

    Args:
        model (torch.nn.Module): The CLIP model.
        topk_v (int): Number of top visual parameters to keep.
        topk_t (int): Number of top textual parameters to keep.
        freeze (set or None): Set of parameter names to freeze.

    """
    freeze = freeze or set()
    grad_params = {}

    def process_grads(named_params, topk, key_prefix):
        grads = {
            f"{key_prefix}.{n}": p.grad.abs().sum().detach() / p.numel()
            for n, p in named_params if p.grad is not None
        }
        grads = dict(sorted(grads.items(), key=lambda item: item[1], reverse=True))
        
        dict_grads = {}
        counter = 0

        for k in grads:
            if k in freeze:
                dict_grads[k] = False
            else:
                dict_grads[k] = counter < topk
                counter += 1
        return dict_grads
            

    grad_params.update(process_grads(model.visual.named_parameters(), topk_v, 'visual'))
    grad_params.update(process_grads(model.transformer.named_parameters(), topk_t, 'transformer'))

    return grad_params


def get_model_optim(arch: str = "RN50", device: str = 'cpu', load_path: str = "", 
                    lr: float = 5e-5) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Load a CLIP model and initialize its optimizer.

    Args:
        arch (str): Model architecture ("RN50" or "ViT-B/16").
        device (str): Device to load model onto.
        load_path (str): Path to load pre-trained weights from (if any).
        lr (float): Learning rate for the optimizer.

    """
    url = clip._MODELS[arch]
    model_path = clip._download(url)
    print("Loading model...")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict()).float().to(device).eval()
    
    if load_path:
        print(f"LOADING FROM {load_path}")
        model.load_state_dict(torch.load(load_path, map_location="cpu"))
        model = model.float().to(device).eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    return model, optimizer


def eval_all_ds(model: torch.nn.Module, datasets_cls: dict, forget_ds: str, forget_lbl: str, 
                all_loaders: dict, train_loader: torch.utils.data.DataLoader or None = None, 
                eval_forgetonly: bool = False, debug: bool = False, device: str = 'cpu') -> dict:
    """
    Evaluate the model across all datasets.

    Args:
        model (torch.nn.Module): The CLIP model to evaluate.
        datasets_cls (dict): Dictionary of dataset classes.
        forget_ds (str): Dataset name to focus forgetting on.
        forget_lbl (str): Label to forget.
        all_loaders (dict): Dictionary of test dataloaders.
        train_loader (DataLoader or None): Training data loader for evaluation.
        eval_forgetonly (bool): Evaluate only the forget dataset if True.
        debug (bool): Enable debug mode.
        device (str): Device.
    """
    results = {ds: {} for ds in all_loaders}
    for ds in all_loaders:
        model.eval()
        test_loader = all_loaders[ds]
        
        classnames = datasets_cls[ds].classnames
        clip_weights = clip_classifier(classnames, [CUSTOM_TEMPLATES[ds]], model).to(device)
        
        if ds == forget_ds:
            cls_acc_test = None
            no_cls_acc = None
            if debug:
                acc, (labels, clip_logits_test) = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=True)
                id_lbl = classnames.index(forget_lbl)
                mask_labels = labels != id_lbl

                cls_acc_test = confusion_matrix(labels, clip_logits_test.argmax(1))[id_lbl]
                cls_acc_test = cls_acc_test[id_lbl] / cls_acc_test.sum()
                    
                no_cls_acc = confusion_matrix(labels[mask_labels], clip_logits_test.argmax(1)[mask_labels])
                no_cls_acc = np.diag(no_cls_acc).sum() / no_cls_acc.sum()
            
            # include accuracy of the train data if not None
            if train_loader is not None:
                acc_train = evaluate_clip_zs(model, train_loader, clip_weights, device=device, out_conf=False)
                acc_train = acc_train 
            else:
                acc_train = None

            results[ds][forget_lbl] = {'cls_acc_test' : cls_acc_test, 
                                      'no_cls_acc' : no_cls_acc, 
                                      'acc_train' : acc_train}

            print(f"{10*'+++'} Train dataset: {ds} - {results[ds][forget_lbl]} {10*'+++'}")

                
        else:
            if eval_forgetonly: continue
            acc = evaluate_clip_zs(model, test_loader, clip_weights, device=device, out_conf=False)
            results[ds]['all'] = {'all_ds' : acc}     
            print(f"{10*'+++'} {ds} - {acc} {10*'+++'}")
    
    return results



class ForgetDataset:
    def __init__(self, data: list or torch.Tensor, list_txt: list, prompt: str = "A picture of {}", 
                 tfm_train: object or None = None, idx_cls_forget: int or None = None, 
                 forget_class: str or None = None, dict_out: bool = False, is_generated: bool = False):
        """
        Custom dataset for handling forget data, either real or generated.

        Args:
            data (list or torch.Tensor): List of image paths or tensor of generated images.
            list_txt (list): List of text labels.
            prompt (str): Prompt template for text tokenization.
            tfm_train (object or None): Transform for training data.
            idx_cls_forget (int or None): Index of the class to forget.
            forget_class (str or None): Class name to forget.
            dict_out (bool): Return data as dictionary if True.
            is_generated (bool): Flag indicating if data is generated.
        """
        self.data = data  # Can be either image paths or tensors
        self.title = {t: clip.tokenize(prompt.format(t)) for t in list_txt}
        self.forget_class = forget_class
        self.tfm_train = tfm_train
        self.dict_out = dict_out
        self.idx_cls_forget = idx_cls_forget
        self.is_generated = is_generated

        # If dealing with generated data, apply final transform only
        if is_generated:
            self.transform = transforms.Compose([tfm_train.transforms[-1]])
        else:
            self.transform = tfm_train  # Use full transform for real images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_generated:
            image = self.transform(deprocess(self.data[idx].unsqueeze(0)))
            title = self.title[self.forget_class].squeeze()
        else:
            cls_name = self.data[idx].classname
            image = self.tfm_train(Image.open(self.data[idx].impath).convert('RGB'))
            title = self.title[cls_name].squeeze()

        if self.dict_out:
            return {'img': image, 'label': torch.tensor(self.idx_cls_forget).long()}
        return image, title
    
    
        
def clip_classifier(classnames: list, template: list, clip_model: torch.nn.Module) -> torch.Tensor:
    """
    Generate CLIP classifier weights for given classnames.

    Args:
        classnames (list): List of class names.
        template (list): List of prompt templates.
        clip_model (torch.nn.Module): Pre-trained CLIP model.

    """
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(clip_model.visual.conv1.weight.device)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1)#.cuda()
    return clip_weights

def cls_acc(output: np.ndarray, target: np.ndarray, topk: int = 1) -> float:
    """
    Compute classification accuracy.

    Args:
        output (np.ndarray): Model output logits.
        target (np.ndarray): True labels.
        topk (int): Number of top predictions to consider.

    """
    # Get the topk predictions
    # pred = np.argsort(output, axis=1)[:, -topk:][:, ::-1].T
    pred = np.argmax(output, axis=1)
    
    # Check if predictions match the target
    correct = pred == target.reshape(1, -1)
    
    # Calculate accuracy
    acc = correct[:topk].reshape(-1).sum(0)
    acc = 100 * acc / target.shape[0]
    
    return acc

def evaluate_clip_zs(model: torch.nn.Module, loader: torch.utils.data.DataLoader, clip_weights: torch.Tensor, 
                     device: str or None = None, out_conf: bool = False, output_probs: bool = False) -> tuple or float:
    """
    Evaluate zero-shot performance of CLIP model.

    Args:
        model (torch.nn.Module): CLIP model to evaluate.
        loader (DataLoader): DataLoader with evaluation data.
        clip_weights (torch.Tensor): Precomputed CLIP classifier weights.
        device (str or None): Device for computation.
        out_conf (bool): Return confusion matrix data if True.
        output_probs (bool): Return probabilities if True.

    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):   
            images = batch['img']
            target = batch['label']

            images, target = images.to(device), target.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.cpu())
            labels.append(target.cpu())
            
    labels = torch.cat(labels)
    features = torch.cat(features)
    
    clip_logits_test = 100. * features @ clip_weights.detach().cpu().numpy()
    acc = cls_acc(clip_logits_test.detach().cpu().numpy(), labels.detach().cpu().numpy())
    acc = acc / 100.
    
    if output_probs:
        probs = torch.nn.functional.softmax(clip_logits_test, dim=-1)
    
    if out_conf:
        if output_probs:
            return acc, (labels, clip_logits_test), probs
            
        return acc, (labels, clip_logits_test)
    
    if output_probs:
        return acc, probs

    return acc

            
def compute_avg_gain(df: pd.DataFrame) -> pd.Series:
    """
    Compute average gain in forgetting performance across datasets.

    Args:
        df (pd.DataFrame): DataFrame with evaluation results.

    """
    forget_perc = 1.- (df['cls_Noforget'] - df['cls_forget'])/df['cls_Noforget']
    list_main_perc = ((df['full_Noforget'] - df['full_forget'])/df['full_Noforget']).clip(0)
    forget_perc_scars = ((df['full_StanfordCars'] - df['res_StanfordCars'])/df['full_StanfordCars']).clip(0).fillna(0)
    forget_perc_caltech = ((df['full_Caltech101'] - df['res_Caltech101'])/df['full_Caltech101']).clip(0).fillna(0)
    forget_perc_oxflow = ((df['full_OxfordFlowers'] - df['res_OxfordFlowers'])/df['full_OxfordFlowers']).clip(0).fillna(0)
    forget_perc_sdogs = ((df['full_StanfordDogs'] - df['res_StanfordDogs'])/df['full_StanfordDogs']).clip(0).fillna(0)
    # divide by 5 as we have 4 datasets + forget_perc (we have 6 elements below but in each row one element is 0 as it's NA)
    scores = (forget_perc + list_main_perc + forget_perc_scars + forget_perc_caltech + forget_perc_oxflow + forget_perc_sdogs)/5
    return scores.astype(float)


def create_results(res_folder, gen_data=True, all_res=False, rn=True, log_name='logs.json',):
    
    with open(f"generated/results_zs_all_RN50.pkl", "rb") as f:
        results_zs_RN = pickle.load(f)  

    with open(f"generated/results_zs_all_ViT16.pkl", "rb") as f:
        results_zs_ViT = pickle.load(f)  
        
    if res_folder != "":
        with open(res_folder + f"/{log_name}", "r") as f:
            all_logs = json.load(f)

        with open(res_folder + "/args.txt", "r") as f:
            args = f.read()
    else:
        all_logs = log_name
        args = ""
        
    if rn:
        results_zs = results_zs_RN
    else:
        results_zs = results_zs_ViT
        
    full_df = []
    final_results = {}
    add_cols = []
    for jj, file in enumerate(all_logs):
        if gen_data:
            cols = ['cls_forget', 'full_forget', 'acc_train',
                        'acc_val_hard', 'acc_val_true_data']
        else:
            cols = ['cls_forget', 'full_forget', 'acc_train']
        
        single_df = pd.DataFrame(columns=cols)

        if file == 'settings': continue
        final_results[file] = {}

        for ii, k in enumerate(all_logs[file]):
            if not all_logs[file][k]: continue
            final_results[file][k] = all_logs[file][k]['final_results'][file][k]
            
            try:
                std = list(all_logs[file][k].keys())[-2]
            except:
                std = 0
           
            single_df.loc[ii, cols] = pd.DataFrame(final_results[file][k].items())[1].values
            single_df.loc[ii, 'name'] = k
            single_df.loc[ii, 'ds'] = file
            
            single_df.loc[ii, 'full_Noforget'] = results_zs[file][k]['no_cls_acc']
            single_df.loc[ii, 'cls_Noforget'] = results_zs[file][k]['cls_acc_test']
            single_df.loc[ii, 'std'] = std
            if all_res:
                for k1 in list(all_logs[file][k]['final_results'].keys()):
                    if 'all' not in all_logs[file][k]['final_results'][k1]: continue
                    single_df.loc[ii, f'res_{k1}'] = all_logs[file][k]['final_results'][k1]['all']['all_ds']
                    single_df.loc[ii, f'full_{k1}'] = results_zs[k1][list(results_zs[k1].keys())[0]]['full_acc']
                    if f'res_{k1}' not in add_cols:
                        add_cols.append(f'res_{k1}')
                    if f'full_{k1}' not in add_cols:
                        add_cols.append(f'full_{k1}')
                
        full_df.append(single_df)
    
    # if not comparables:
    full_df = pd.concat(full_df)[['ds', 'name', 'std', 'full_Noforget', 'cls_Noforget'] + cols + add_cols]
    # else:
    all_logs['settings'] = ""
    # full_df = pd.concat(full_df)[['ds', 'name', 'full_Noforget', 'cls_Noforget'] + cols + add_cols]

    if gen_data:
        full_df = full_df.drop('acc_train', axis=1)
        
    # if return_logs:
    #     return full_df, args, all_logs['settings'], all_logs
    return full_df, args
            
    
    
def freeze_batch_norm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
                
                
def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        # T.ToPILImage(),
    ])
    return transform(img)


    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor, return_second=False):
        _max = tensor.max()
        _min = tensor.min()
        tensor1 = tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        tensor1 = torch.clamp(tensor1, min=_min, max=_max)
        if return_second:
            tensor2 = tensor + 2 * torch.randn(tensor.size()).to(self.device) * self.std + self.mean
            tensor2 = torch.clamp(tensor2, min=_min, max=_max)
            return tensor1, tensor2
        return tensor1
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)