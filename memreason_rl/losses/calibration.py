import torch

def gaussian_nll(mean,logvar,target,mask):
    if not mask.any():
        return mean.sum()*0
    mean,logvar,target=mean[mask],logvar[mask].clamp(-8,6),target[mask]
    return .5*(torch.log(mean.new_tensor(2*torch.pi))+logvar+(target-mean).square()/logvar.exp()).mean()
