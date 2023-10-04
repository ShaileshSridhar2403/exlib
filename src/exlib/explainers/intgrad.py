import torch
import torch.nn.functional as F
from tqdm import tqdm
from .common import FeatureAttrOutput

def intgrad_image_class_loss_fn(y, label):
    assert y.ndim == 2   # (N, num_classes)
    assert len(label) == y.size(0)
    loss = 0.0
    for i, l in enumerate(label):
        loss += y[i,l]
    return loss


def intgrad_image_seg_loss_fn(y, label):
    assert y.ndim == 4  # (N, num_segs, H, W)
    assert len(label) == y.size(0)
    loss = 0.0
    for i, l in enumerate(label):
        inds = y.argmax(dim=1) # Max along the channels
        H = F.one_hot(inds, num_classes=y.size(1))
        H = H.unsqueeze(1).transpose(1,-1).view(y.shape)
        L = (y * H).flatten(1).sum(dim=1)
        return L


# Do classification 
def explain_image_with_intgrad(X, model, loss_fn,
                               X0 = None,
                               num_steps = 32,
                               progress_bar = False,
                               postprocess = None):
    """
    Explain a classification model with Integrated Gradients.
    """
    # Default baseline is zeros
    X0 = torch.zeros_like(X) if X0 is None else X0
    if X0 is None:
        X0 = torch.zeros_like(X)

    acc = torch.zeros_like(X)

    pbar = tqdm(range(num_steps)) if progress_bar else range(num_steps)
    for k in pbar:
        ak = k / num_steps
        Xk = X0 + ak * (X - X0)
        Xk.requires_grad_()
        y = model(Xk)
        if postprocess:
            y = postprocess(y)

        loss = loss_fn(y)
        loss.sum().backward()
        acc += Xk.grad / num_steps # Recall that step_size = 1 / num_steps

    return FeatureAttrOutput(acc, {})


