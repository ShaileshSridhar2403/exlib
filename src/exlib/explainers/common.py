from collections import namedtuple

import torch
import torch.nn.functional as F

AttributionOutput = namedtuple("AttributionOutput", ["attributions", "explainer_output"])

def patch_segmenter(image, sz=(8,8)): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    X = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=X.size()[:2], mode='nearest').long()
    segments = segments[0,0].numpy()
    return segments
