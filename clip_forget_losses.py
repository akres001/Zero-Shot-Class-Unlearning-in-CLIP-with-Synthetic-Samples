import torch
from copy import deepcopy
from clip import clip
from typing import List


def clip_classifier(classnames: List[str], template: List[str], clip_model: torch.nn.Module) -> torch.Tensor:
    """
    Generate CLIP classifier weights for a set of class names using a prompt template.

    Args:
        classnames (List[str]): List of class names to generate embeddings for.
        template (List[str]): List of prompt templates with a placeholder for the class name
        clip_model (torch.nn.Module): CLIP model instance.

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
 

def lipschitz_loss_joint(model: torch.nn.Module, text: torch.Tensor, image: torch.Tensor, 
                         transform: callable, n_samples: int = 5, device: str = 'cpu', 
                         verbose: bool = False) -> torch.Tensor:
    """
    Compute a joint Lipschitz loss between text and image embeddings under Gaussian noise perturbation.
    Since text tokens are discrete and cannot be perturbed directly with Gaussian noise, the loss compares
    text embeddings to noisy image embeddings, leveraging their shared projection space in CLIP.

    Args:
        model (torch.nn.Module): CLIP model
        text (torch.Tensor): Tokenized text input tensor
        image (torch.Tensor): Input image tensor 
        transform (callable): Function that applies Gaussian noise 
        n_samples (int, optional): Number of noisy image samples to generate for averaging the loss.
        device (str, optional): Device
        verbose (bool, optional): Verbose parameter. 
    
    """

    out_txt = model.encode_text(text)  
    out_img = model.encode_image(image)   
        
    loss = torch.tensor(0.0, device=device)
    out_n = torch.tensor(0.0, device=device)
    in_n = torch.tensor(0.0, device=device)
    
    for _ in range(n_samples):
        image2 = transform(deepcopy(image))             
        with torch.no_grad():
            out2 = model.encode_image(image2)
            
        flatimg, flatimg2 = image.view(image.size()[0], -1), image2.view(image2.size()[0], -1)

        in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)     
        
        out_norm_txt = torch.linalg.vector_norm(out_txt - out2, dim=1)
        out_norm_img = torch.linalg.vector_norm(out_img - out2, dim=1)
        
        out_norm = (out_norm_txt + out_norm_img)
        in_norm = in_norm * 2
        in_n += (in_norm.sum() * 2)
        out_n += (out_norm_txt.sum() + out_norm_img.sum())
        K = ((out_norm / in_norm).sum()).abs()#pow(2)#  0.1                                               
        loss += K
    
    loss /= (n_samples * 2)
    in_n /= (n_samples * 2) 
    out_n /= (n_samples * 2)
    if verbose:
        print(f"Loss {loss}, out diff {out_n}, noise {in_n}")  
    return loss
