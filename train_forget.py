import os
from tqdm import tqdm
import json
import copy
import torch.nn as nn
import torch.optim as optim 
import argparse

from clip import clip
import clip_forget_utils
from clip_forget_utils import *
from frozen_params import *
from train_utils import *
from forget_cls import *

import clip_forget_utils
clip_forget_utils.CUSTOM_TEMPLATES = CUSTOM_TEMPLATES

torch.set_num_threads(5)

device = 'cuda'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/result0", help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="only positive value enables a fixed seed")    
    parser.add_argument("--generated_data", type=int, default=1, help="use spectral adapter or not")
    parser.add_argument("--save_best_model", type=int, default=1, help="whether to normalize embeddings")
    parser.add_argument("--debug", type=int, default=1, help="debug computing all results")
    parser.add_argument("--max_epochs", type=int, default=30, help="use spectral adapter or not")
    parser.add_argument("--min_train_acc_stop", type=float, default=0.2, help="no need to reach accuracy of 0")
    parser.add_argument("--n_samples", type=int, default=10, help="use spectral adapter or not")
    parser.add_argument("--run_ds", type=str, default="")
    parser.add_argument("--backbone_arch", type=str, default="RN50")    
    
    args = parser.parse_args()
    args.config_file = "configs/trainers/adam_lr2e-4_B256_ep200_ViT16.yaml"
    print("Arguments : ", args)

    cfg = initialize_config(args)
    test_datasets, test_dataloaders, datasets_cls, tfm_train, tfm_test = load_test_datasets(cfg)
    os.makedirs(args.output_dir, exist_ok=True)   
    
    dataset = None
        
    MODEL_ARCH = "ViTB16" if args.backbone_arch == "ViT-B/16" else args.backbone_arch
     
    if args.generated_data:
        dataloader_generated = load_generated_data(MODEL_ARCH)
    
    results_zs = load_results(args)
    init_topvis, init_toptext = set_hyperparameters(args)
    textparam_schedule, imgparam_schedule, std_scheduler = set_param_schedules(init_topvis, init_toptext, args.backbone_arch)
    set_seeds(args.seed)

    num_epochs = args.max_epochs
    n_samples = args.n_samples
    MAX_STEPS = 100 
    BATCH_SIZE = 64
       
    all_logs = {}
    
    if args.run_ds != "" :
        args.run_ds = [item for item in args.run_ds.split(',')]
        assert all([ds in all_ds for ds in args.run_ds])
        run_ds = args.run_ds
    else:
        run_ds = all_ds[:]
        
    with open(args.output_dir + "/args.txt", "w") as f:
        f.write(str(args)) 
    
    output_base = args.output_dir
    
    for main_ds in run_ds:
        # print("STD", std_scheduler[0])        
        args.output_dir = output_base + f"/{main_ds}"
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"main ds: {main_ds}")

        main_dataset_classes = datasets_cls[main_ds]

        all_logs[main_ds] = {}
        all_logs['settings'] = { 'n_samples' : n_samples, 'BATCH_SIZE' : BATCH_SIZE}
       
        best_model_state = None

        for forget_label in forget_classes_all[main_ds]: 
        
            print(f"forget_label {forget_label}")
            all_logs[main_ds][forget_label] = {}

            idx_cls_forget = main_dataset_classes.classnames.index(forget_label)
            if args.generated_data:

                if forget_label not in dataloader_generated[main_ds]:
                    print("SKIP forget label", forget_label, dataloader_generated[main_ds].keys())
                    continue 
                    
                gen_examples = dataloader_generated[main_ds][forget_label]
                
                train_ds_generated = ForgetDataset(gen_examples, [forget_label], forget_class=forget_label, tfm_train=tfm_train, is_generated=True)
                train_loaded_generated = torch.utils.data.DataLoader(train_ds_generated, batch_size=BATCH_SIZE, shuffle=False) 

                train_ds_generated_foreval = ForgetDataset(gen_examples, [forget_label], forget_class=forget_label, 
                                                                tfm_train=tfm_train, idx_cls_forget=idx_cls_forget, 
                                                                dict_out=True, is_generated=True)
                train_loader_eval = torch.utils.data.DataLoader(train_ds_generated_foreval, batch_size=32, shuffle=False)
                train_loader = train_loaded_generated
              
                relevant_valid = ds_relevant_forget(main_dataset_classes, forget_label, data_type='valid') # for_eval=True)
                valid_ds_eval_true = ForgetDataset(relevant_valid, [forget_label], #forget_class=forget_label, 
                                                   idx_cls_forget=idx_cls_forget, tfm_train=tfm_test, 
                                                   dict_out=True, is_generated=False)
                valid_loader_eval_true = torch.utils.data.DataLoader(valid_ds_eval_true, batch_size=32, shuffle=False) 


            else:
                relevant_trainx = ds_relevant_forget(main_dataset_classes, forget_label)

                train_ds = ForgetDataset(relevant_trainx, [forget_label], forget_class=forget_label, tfm_train=tfm_train,  is_generated=False)
                train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) 
                
                train_ds_eval = ForgetDataset(relevant_trainx, [forget_label], forget_class=forget_label, 
                                              idx_cls_forget=idx_cls_forget, tfm_train=tfm_test, 
                                              dict_out=True, is_generated=False)
                train_loader_eval = torch.utils.data.DataLoader(train_ds_eval, batch_size=32, shuffle=False) 
                valid_loader_eval_true = None
                

            model, optimizer = get_model_optim(arch=args.backbone_arch,device=device)

            std_idx = 0
            best_train_acc = 1.
            next_std = std_scheduler[std_idx]
            TOPTEXT = textparam_schedule[std_idx]
            TOPVIS = imgparam_schedule[std_idx]

            transform_gaussian = AddGaussianNoise(std=std_scheduler[std_idx], device=device)

            print(f"{20*'='} ZS for label {forget_label}", results_zs[main_ds][forget_label], f"{20*'='}")

            acc_step = min(len(train_loader), 5)
            for epoch in range(args.max_epochs):
                last_model_state = copy.deepcopy(model.state_dict())
                last_optim_state = copy.deepcopy(optimizer.state_dict())

                for ii, batch in enumerate(train_loader):
                    params_before_update = copy.deepcopy(model.state_dict())

                    loss = train_epoch(batch, model, transform_gaussian, n_samples, device=device)
                    loss = loss / acc_step
                    loss.backward()
                    
                    freeze = FREEZE_RN if args.backbone_arch == "RN50" else None
                        
                    imp_params = get_important_params_mask(model, topk_v = TOPVIS, topk_t = TOPTEXT, freeze=freeze)
                    true_params = [k for k, v in imp_params.items() if v] # FORCE_UPDATE

                    # freeze all other params
                    for n, p in model.named_parameters():
                        if n not in true_params:
                            p.grad = None
              
                    if ((ii + 1) % acc_step == 0) or (len(train_loader) == ii + 1):
                        optimizer.step()
                        optimizer.zero_grad()

                    # If # batches > 1 we need to check at every update!      
                    with torch.no_grad():
                        clip_weights = clip_classifier(main_dataset_classes.classnames, [CUSTOM_TEMPLATES[main_ds]], model).to(device)
                        acc_train = evaluate_clip_zs(model, train_loader_eval, clip_weights, device=device, out_conf=False)
                        if acc_train <= args.min_train_acc_stop:
                            # print("BROKE EARLIER")
                            # print("acc_train", acc_train)
                            break

                model.eval()
                    
                results_ds, acc = evaluate_and_log(args, model, datasets_cls, main_ds, forget_label, test_dataloaders, train_loader_eval, valid_loader_eval_true,
                                                   all_logs,  classnames=main_dataset_classes.classnames, debug=args.debug, eval_forgetonly=True)

                print(f"best accuracy, {best_train_acc},  current acc  {acc}")
                print(results_ds[main_ds][forget_label])

                if acc < best_train_acc:
                    best_train_acc = acc

                    all_logs[main_ds][forget_label][str(std_scheduler[std_idx])] = results_ds[main_ds][forget_label]

                    best_model_state = copy.deepcopy(model.state_dict())
                    if args.save_best_model:
                        torch.save(best_model_state, args.output_dir + f"/model_{main_ds}_{forget_label}.pth")

                    print(f"{20 * '*'} Saved model results", results_ds[main_ds][forget_label])
                else:
                    # restore previous version if no decrease in accuracy
                    model_last_state_load = copy.deepcopy(model.state_dict())
                    
                    model.load_state_dict(last_model_state)
                    optimizer.load_state_dict(last_optim_state)

                    std_idx += 1

                    if std_idx == len(std_scheduler) or std_idx >= MAX_STEPS: 
                        break

                    TOPTEXT = textparam_schedule[std_idx]
                    TOPVIS = imgparam_schedule[std_idx]
                    next_std = std_scheduler[std_idx]

                    transform_gaussian = AddGaussianNoise(std=next_std, device=device)

                if acc <= args.min_train_acc_stop:
                    break
                    
                        
            print("Computing results")                
            if best_model_state is None:
                best_model_state = model_last_state_load

            model.load_state_dict(best_model_state)

            results_ds, acc = evaluate_and_log(args, model, datasets_cls, main_ds, forget_label, test_dataloaders, train_loader_eval, valid_loader_eval_true,
                                               all_logs, classnames=main_dataset_classes.classnames, debug=True, eval_forgetonly=False)

            all_logs[main_ds][forget_label]['final_results'] = results_ds
            print(f"{20 * '*'} Final results for {forget_label}", results_ds[main_ds][forget_label])
            
            best_model_state = None
            
        print(all_logs)
        

    with open(output_base + "/logs.json", "w") as f:
        json.dump(all_logs, f)
        
    while True:
        import time
        time.sleep(1000)
        print("slept")
        pass