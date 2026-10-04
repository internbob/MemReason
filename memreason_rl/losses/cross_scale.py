from torch.nn import functional as F

def cross_scale_consistency(state_a,state_b):
    return F.mse_loss(state_a,state_b)
