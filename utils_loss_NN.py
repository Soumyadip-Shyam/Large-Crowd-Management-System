import torch
import torch.nn.functional as F

def MSE_BCE(y_true, y_pred, alpha=1000, beta=10):
    """
    Custom loss function combining MSE and Binary Cross Entropy
    
    Args:
        y_true: Ground truth tensor
        y_pred: Predicted tensor  
        alpha: Weight for MSE loss (default: 1000)
        beta: Weight for BCE loss (default: 10)
    
    Returns:
        Combined loss value
    """
    mse = torch.mean(torch.square(y_true - y_pred), dim=-1)
    bce = torch.mean(F.binary_cross_entropy(y_pred, y_true, reduction='none'), dim=-1)
    return alpha * mse + beta * bce

# Alternative implementation using MSE loss function directly
def MSE_BCE_alt(y_true, y_pred, alpha=1000, beta=10):
    """Alternative implementation using F.mse_loss"""
    mse = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=-1)
    bce = F.binary_cross_entropy(y_pred, y_true, reduction='none').mean(dim=-1)
    return alpha * mse + beta * bce

# Usage as a loss function class (recommended for training)
class MSE_BCE_Loss(torch.nn.Module):
    def __init__(self, alpha=1000, beta=10):
        super(MSE_BCE_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        mse = torch.mean(torch.square(y_true - y_pred), dim=-1)
        bce = torch.mean(F.binary_cross_entropy(y_pred, y_true, reduction='none'), dim=-1)
        return self.alpha * mse + self.beta * bce
