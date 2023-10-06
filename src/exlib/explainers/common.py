from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FamWrapper(nn.Module):
    """ Wrap a model with a pre/post processing function
    """
    def __init__(self, model, preprocessor=None, postprocessor=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def forward(self, x):
        if self.preprocessor:
            x = self.preprocessor(x)

        y = self.model(x)

        if self.postprocessor:
            y = self.postprocessor(y)

        return y


FeatureAttrOutput = namedtuple("FeatureAttrOutput", ["attributions", "explainer_output"])


class FeatureAttrMethod(nn.Module): 
    """ Explaination methods that create feature attributions should follow 
    this signature. """
    def __init__(self, model): 
        super().__init__() 
        self.model = model

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()


class Seg2ClsWrapper(nn.Module):
    """ Simple wrapper for converting to be classification compatible.
    We are assuming

        (N,C,H,W) --[cls model]--> (N,K)
        (N,C,H,W) --[seg model]--> (N,K,H,W)

        (N,C,H,W) --[wrapd seg]--> (N,K)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(x)
        N, K, H, W = y.shape
        A = y.argmax(dim=1)             # (N,H,W)
        B = F.one_hot(A, num_classes=K) # (N,H,W,K)
        C = B.permute(0,3,1,2)          # (N,K,H,W)
        D = C.view(N,K,1,H,W) * y.view(N,1,K,H,W)
        L = D.sum(dim=(2,3,4))
        return L


def patch_segmenter(image, sz=(8,8)): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    x = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=x.size()[:2], mode='nearest').long()
    segments = segments[0,0].numpy()
    return segments

def torch_img_to_np(x): 
    if x.dim() == 4: 
        return x.permute(0,2,3,1).numpy()
    elif x.dim() == 3: 
        return x.permute(1,2,0).numpy()
    else: 
        raise ValueError("Image tensor doesn't have 3 or 4 dimensions")

def np_to_torch_img(x_np):
    x = torch.from_numpy(x_np) 
    if x.dim() == 4: 
        return x.permute(0,3,1,2)
    elif x.dim() == 3: 
        return x.permute(2,0,1)
    else: 
        raise ValueError("Image array doesn't have 3 or 4 dimensions")

