import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .common import Evaluator, convert_idx_masks_to_bool


class CompSuff(Evaluator):

    def __init__(self, model, 
                 mode: str, 
                 k_fraction: float, 
                 postprocess=None,
                 task_type='cls'):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'comp' or 'suff'.
            k_fraction (float): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(CompSuff, self).__init__(model, postprocess)
        
        assert mode in ['comp', 'suff']
        assert task_type in ['cls', 'reg']
        self.mode = mode
        self.task_type = task_type
        self.k_fraction = k_fraction

    def forward(self, X, Z, kwargs=None, verbose=0, save_to=None):
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
        self.model.eval()
        img_tensor = X
        explanation = Z
        if len(X.shape) == 4: # image
            bsz, n_channel, img_dim1, img_dim2 = X.shape
            HW = img_dim1 * img_dim2
            model_type = 'image'
        else: # text
            bsz, seq_len = X.shape
            HW = seq_len
            n_channel = 1
            model_type = 'text'
        with torch.no_grad():
            if kwargs:
                pred = self.model(img_tensor, **kwargs)
            else:
                pred = self.model(img_tensor)
            if self.postprocess is not None:
                pred = self.postprocess(pred)
        pred = torch.softmax(pred, dim=-1)
        top, c = torch.max(pred, 1)
        if self.k_fraction < 1:
            step = int(self.k_fraction * HW)
        else:
            step = int(self.k_fraction)

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
        with torch.no_grad():
            if kwargs:
                pred_mod = self.model(start, **kwargs)
            else:
                pred_mod = self.model(start)
            if self.postprocess is not None:
                pred_mod = self.postprocess(pred_mod)
        pred_mod = torch.softmax(pred_mod, dim=-1)
        scores = pred[range(bsz), c] - pred_mod[range(bsz), c]

        return scores
    
class CompSuffSem(CompSuff):

    def __init__(self, model, 
                 mode: str,
                 k_fraction: float, 
                 postprocess=None):
        """Create deletion/insertion metric instance. by semantic partitions
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'comp' or 'suff'.
            k_fraction (float): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super().__init__(model, mode, k_fraction, postprocess)

    def forward(self, X, Z, sem_part, kwargs=None):
        """Run metric on one image-saliency pair.
            Args:
                X = img_tensor (Tensor): normalized image tensor. (bsz, n_channel, img_dim1, img_dim2)
                Z = explanation (Tensor): saliency map. (bsz, 1, img_dim1, img_dim2)
                sem_part = semantic partition (Tensor): int (bsz, img_dim1, img_dim2)
                    Partition by SOP masks.
                verbose (int): in [0, 1, 2].
                    0 - return list of scores.
                    1 - also plot final step.
                    2 - also plot every step and print 2 top classes.
                save_to (str): directory to save every step plots to.
            Return:
                scores (Tensor): Array containing scores at every step.
        """
        self.model.eval()
        scores_all = []
        for b_i in range(X.size(0)):
            sem_part_bool = convert_idx_masks_to_bool(sem_part[b_i:b_i+1])
            num_masks = sem_part_bool.size(0)

            img_tensor = X[b_i:b_i+1]
            explanation = Z[b_i:b_i+1].to(img_tensor.device)
            if len(X.shape) == 4:
                bsz, n_channel, img_dim1, img_dim2 = img_tensor.shape
                HW = img_dim1 * img_dim2
                model_type = 'image'
            else:
                bsz, seq_len = img_tensor.shape
                HW = seq_len
                n_channel = 1
                model_type = 'text'
            with torch.no_grad():
                if kwargs:
                    pred = self.model(img_tensor, **kwargs)
                else:
                    pred = self.model(img_tensor)
                if self.postprocess is not None:
                    pred = self.postprocess(pred)
            pred = torch.softmax(pred, dim=-1)
            if self.task_type == 'cls':
                top, c = torch.max(pred, 1)
            else:
                c = torch.arange(pred.shape[-1])

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

            t_r_masks = (explanation * sem_part_bool.unsqueeze(1).float()).reshape(num_masks, 
                                                                                -1).mean(-1)
            salient_order_masks = torch.argsort(t_r_masks, dim=-1).flip(-1)
            mask_sem_best = sem_part_bool[salient_order_masks[0]]

            if model_type == 'image':
                start[0,:,mask_sem_best] = finish[0,:,mask_sem_best]
            else:
                start[0,mask_sem_best] = finish[0,mask_sem_best]

            with torch.no_grad():
                if kwargs:
                    pred_mod = self.model(start, **kwargs)
                else:
                    pred_mod = self.model(start)
                if self.postprocess is not None:
                    pred_mod = self.postprocess(pred_mod)
            if self.task_type == 'cls':
                pred_mod = torch.softmax(pred_mod, dim=-1)
                scores = pred[range(bsz), c] - pred_mod[range(bsz), c]
            else:
                criterion = nn.MSELoss(reduction='none')
                mod_loss = criterion(pred_mod, pred)
                scores = mod_loss
            scores_all.append(scores)

        scores_all = torch.cat(scores_all, dim=0)
        return scores_all