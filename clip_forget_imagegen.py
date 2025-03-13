import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import torchvision.transforms as T
import torch
import os

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

torch.manual_seed(1)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        # T.ToPILImage(),
    ])
    return transform(img)

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.
    
    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes
    
    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        # T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def generate_images(model, clip_weights, dataset_name, id_lbl, name_lbl, tfm_test, device, 
                    verbose=False, save_path= "/generated_data/", **kwargs):
    
    save_dir = f"{save_path}/{dataset_name}"
    save_tensor_dir = save_dir + f"/{name_lbl}.pt"
    os.makedirs(save_dir, exist_ok=True)
    
    print(save_tensor_dir)
    if os.path.exists(save_tensor_dir):
        print(save_tensor_dir, " exists")
        return

    print("KWARGS", kwargs)
    
    l2_reg = kwargs.pop('l2_reg', 1e-3)
    learning_rate = kwargs.pop('learning_rate', 25)
    num_iterations = kwargs.pop('num_iterations', 3000)
    blur_every = kwargs.pop('blur_every', 10)
    max_jitter = kwargs.pop('max_jitter', 16)
    n_generated = kwargs.pop('n_generated', 64)
    high_prob = kwargs.pop('high_prob', 0.8)
    sub_prob = kwargs.pop('sub_prob', 0.025)
    
    all_images = []
    FOUND = False
    FIRST_ITER = True
    while len(all_images) < n_generated:
        if not FOUND:
            if not FIRST_ITER:
                high_prob = max(high_prob-sub_prob, 0.45)
            
        FIRST_ITER = False
        
        if verbose and len(all_images) % 10 == 0:
            print(len(all_images))
        # Randomly initialize the image as a PyTorch Tensor, and make it requires gradient.
        img = torch.randn(1, 3, 224, 224).mul_(1.0).float().to(device).requires_grad_()

        for t in range(num_iterations):
            # Randomly jitter the image a bit; this gives slightly nicer results
            ox, oy = random.randint(0, max_jitter), random.randint(0, max_jitter)
            img.data.copy_(jitter(img.data, ox, oy))

            image_features = model.encode_image(tfm_test.transforms[-1](img))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_per_image = model.logit_scale.exp() * image_features @ clip_weights
            # break

            target_score = logits_per_image[:, id_lbl] - l2_reg * torch.norm(img)
            target_score.backward()
            with torch.no_grad():
                img += learning_rate * img.grad / torch.norm(img.grad)
                img.grad.zero_()

            # Undo the random jitter
            img.data.copy_(jitter(img.data, -ox, -oy))

            # As regularizer, clamp and periodically blur the image
            for c in range(3):
                lo = float(-SQUEEZENET_MEAN[c] / SQUEEZENET_STD[c])
                hi = float((1.0 - SQUEEZENET_MEAN[c]) / SQUEEZENET_STD[c])
                img.data[:, c].clamp_(min=lo, max=hi)
            if t % blur_every == 0:
                blur_image(img.data, sigma=0.5)

            
            img_test = deprocess(img.data.clone()).unsqueeze(0)
            # at the iteration when model predicts the class we stop
            with torch.no_grad():
                image_features  = model.encode_image(tfm_test.transforms[-1](img_test))
                image_features /= image_features.norm(dim=-1, keepdim=True)

                clip_logits_test = 100. * image_features.detach().cpu() @ clip_weights.detach().cpu()

                if clip_logits_test.argmax() == id_lbl:
                    probs = torch.nn.functional.softmax(clip_logits_test)
                    
                    if probs[0, id_lbl] > high_prob:
                        all_images.append(img)
                        print(probs.argmax(), id_lbl, clip_logits_test[0, id_lbl], probs[0, id_lbl])
                        print(f"break after {t} iterations")
                        FOUND = True
                        break
                        
                            
        if t == (num_iterations-1):
            FOUND = False

        
    all_images = torch.cat(all_images)
    print("all_images shape", all_images.shape)
    torch.save(all_images.detach(), save_tensor_dir)
            
            
