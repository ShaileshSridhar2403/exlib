import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .common import Evaluator

class CompSuff(Evaluator):

    def __init__(self, model, 
                 mode: str, 
                 k_fraction: float, 
                 postprocess=None):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'comp' or 'suff'.
            k_fraction (float): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(CompSuff, self).__init__(model, postprocess)
        
        assert mode in ['comp', 'suff']
        self.mode = mode
        self.k_fraction = k_fraction

    def forward(self, X, Z, verbose=0, save_to=None, return_dict=False):
        """Run metric on one image-saliency pair.
            Args:
                X = img_tensor (Tensor): normalized image tensor. (bsz, n_channel, img_dim1, img_dim2)
                Z = explanation (Tensor): saliency map. (bsz, 1, img_dim1, img_dim2)
                verbose (int): in [0, 1, 2].
                    0 - return list of scores.
                    1 - also plot final step.
                    2 - also plot every step and print 2 top classes.
                save_to (str): directory to save every step plots to.
            Return:
                scores (Tensor): Array containing scores at every step.
        """
        img_tensor = X
        explanation = Z
        bsz, n_channel, img_dim1, img_dim2 = X.shape
        HW = img_dim1 * img_dim2
        pred = self.model(img_tensor)
        if self.postprocess is not None:
            pred = self.postprocess(pred)
        pred = torch.softmax(pred, dim=-1)
        top, c = torch.max(pred, 1)
        # c = c.cpu().numpy()[0]
        # n_steps = (HW + self.step - 1) // self.step
        step = int(self.k_fraction * HW)

        if self.mode == 'comp':
            start = img_tensor.clone()
            finish = torch.zeros_like(img_tensor)
        elif self.mode == 'suff':
            start = torch.zeros_like(img_tensor)
            finish = img_tensor.clone()

        start[start < 0] = 0.0
        start[start > 1] = 1.0
        finish[finish < 0] = 0.0
        finish[finish > 1] = 1.0

        # scores = torch.empty(bsz, n_steps + 1).cuda()
        # Coordinates of pixels in order of decreasing saliency
        t_r = explanation.reshape(bsz, -1, HW)
        salient_order = torch.argsort(t_r, dim=-1)
        salient_order = torch.flip(salient_order, [1, 2])
        if self.mode == 'suff':
            coords_top_k = salient_order[:, :, :step]
            coords = coords_top_k
        else:
            coords_bottom_k = salient_order[:, :, HW - 1 - step:]
            coords = coords_bottom_k
        batch_indices = torch.arange(bsz).view(-1, 1, 1).to(coords.device)
        channel_indices = torch.arange(n_channel).view(1, -1, 1).to(coords.device)

        start.reshape(bsz, n_channel, HW)[batch_indices, 
                                          channel_indices, 
                                          coords] = finish.reshape(bsz, n_channel, HW)[batch_indices,
                                                                                       channel_indices, 
                                                                                       coords]
        
        pred_mod = self.model(start)
        if self.postprocess is not None:
            pred_mod = self.postprocess(pred_mod)
        pred_mod = torch.softmax(pred_mod, dim=-1)
        scores = pred[range(bsz), c] - pred_mod[range(bsz), c]

        return scores