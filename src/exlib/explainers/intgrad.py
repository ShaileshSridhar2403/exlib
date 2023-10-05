import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .common import *


def intgrad_image_class_loss_fn(y, label):
    N, K = y.shape
    assert len(label) == N
    # Make sure the dtype is right otherwise loss will be like all zeros
    loss = torch.zeros_like(label, dtype=y.dtype)
    for i, l in enumerate(label):
        loss[i] = y[i,l]
    return loss


def intgrad_image_seg_loss_fn(y, label):
    N, K, H, W = y.shape
    assert len(label) == N
    loss = torch.zeros_like(label, dtype=y.dtype)
    for i, l in enumerate(label):
        yi = y[i]
        inds = yi.argmax(dim=0) # Max along the channels
        H = F.one_hot(inds, num_classes=K)  # (H,W,K)
        H = H.permute(2,0,1)   # (K,H,W)
        L = (yi * H).sum()
        loss[i] = L
    return loss


# Do classification-based thing
def explain_image_with_intgrad(X, model, loss_fn,
                               X0 = None,
                               num_steps = 32,
                               progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    # Default baseline is zeros
    X0 = torch.zeros_like(X) if X0 is None else X0

    step_size = 1 / num_steps
    intg = torch.zeros_like(X)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        Xk = X0 + ak * (X - X0)
        Xk.requires_grad_()
        y = model(Xk)

        loss = loss_fn(y)
        loss.sum().backward()
        intg += Xk.grad * step_size

    return FeatureAttrOutput(intg, {})


def explain_cls_with_intgrad(model, X, label,
                             X0 = None,
                             num_steps = 32,
                             progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert X.size(0) == len(label)

    # Default baseline is zeros
    X0 = torch.zeros_like(X) if X0 is None else X0

    step_size = 1 / num_steps
    intg = torch.zeros_like(X)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        Xk = X0 + ak * (X - X0)
        Xk.requires_grad_()
        y = model(Xk)

        loss = 0.0
        for i, l in enumerate(label):
            loss += y[i, l]

        loss.backward()
        intg += Xk.grad * step_size

    return FeatureAttrOutput(intg, {})



class IntGradImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model):
        super().__init__(model)

    def forward(self, X, T, **kwargs):
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T)

        return explain_cls_with_intgrad(self.model, X, T, **kwargs)



class IntGradImageSeg(FeatureAttrMethod):
    """ Image segmentation with integrated gradients.
    For this we convert the segmentation model into a classification model.
    """
    def __init__(self, model):
        super().__init__(model)

        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, X, T, **kwargs):
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T)

        return explain_cls_with_intgrad(self.cls_model, X, T, **kwargs)



