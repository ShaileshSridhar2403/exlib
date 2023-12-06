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
def explain_image_with_intgrad(x, model, loss_fn,
                               x0 = None,
                               num_steps = 32,
                               progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros_like(x)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)

    for k in pbar:
        ak = k * step_size
        xk = x0 + ak * (x - x0)
        xk.requires_grad_()
        y = model(xk)

        loss = loss_fn(y)
        loss.sum().backward()
        intg += xk.grad * step_size

    return FeatureAttrOutput(intg, {})


def explain_cls_with_intgrad(model, x, label,
                             x0 = None,
                             num_steps = 32,
                             progress_bar = False):
    """
    Explain a classification model with Integrated Gradients.
    """
    assert x.size(0) == len(label)

    # Default baseline is zeros
    x0 = torch.zeros_like(x) if x0 is None else x0

    step_size = 1 / num_steps
    intg = torch.zeros_like(x)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k * step_size
        xk = x0 + ak * (x - x0)
        xk.requires_grad_()
        y = model(xk)

        loss = 0.0
        for i, l in enumerate(label):
            loss += y[i, l]

        loss.backward()
        intg += xk.grad * step_size

    return FeatureAttrOutput(intg, {})


class IntGradImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model):
        super().__init__(model)

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        with torch.enable_grad():
            return explain_cls_with_intgrad(self.model, x, t, **kwargs)



class IntGradImageSeg(FeatureAttrMethod):
    """ Image segmentation with integrated gradients.
    For this we convert the segmentation model into a classification model.
    """
    def __init__(self, model):
        super().__init__(model)

        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, x, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        with torch.enable_grad():
            return explain_cls_with_intgrad(self.cls_model, x, t, **kwargs)

