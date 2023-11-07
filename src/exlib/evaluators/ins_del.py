import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .common import Evaluator, convert_idx_masks_to_bool

class InsDelCls(Evaluator):

    def __init__(self, model, mode='del', step=224, substrate_fn=torch.zeros_like, 
                 postprocess=None):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(InsDelCls, self).__init__(model, postprocess)
        
        assert mode in ['del', 'ins']
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def auc(self, arr):
        """Returns normalized Area Under Curve of the array."""
        # return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
        # return (arr.sum(-1).sum(-1) - arr[1] / 2 - arr[-1] / 2) / (arr.shape[1] - 1)
        # if len(arr.shape) == 2:
        return (arr.sum(-1) - arr[:, 0] / 2 - arr[:, -1] / 2) / (arr.shape[1] - 1)
        # else:
        #     return (arr.sum(-2) - arr[:, 0] / 2 - arr[:, -2] / 2) / (arr.shape[1] - 1)

    def forward(self, X, Z, kwargs={}, return_dict=False):
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

        bsz, n_channel, img_dim1, img_dim2 = X.shape
        HW = img_dim1 * img_dim2

        with torch.no_grad():
            pred = self.model(img_tensor, **kwargs)
            if self.postprocess is not None:
                pred = self.postprocess(pred)
        top, c = torch.max(pred, 1)

        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()
        start_clone = start.clone()
        # import pdb
        # pdb.set_trace()

        start[start < 0] = 0.0
        start[start > 1] = 1.0
        finish[finish < 0] = 0.0
        finish[finish > 1] = 1.0
        all_states = []

        scores = torch.zeros(bsz, n_steps + 1).cuda()
        
        # Coordinates of pixels in order of decreasing saliency
        t_r = explanation.reshape(bsz, -1, HW)
        salient_order = torch.argsort(t_r, dim=-1)
        salient_order = torch.flip(salient_order, [1, 2])

        for i in tqdm(range(n_steps+1)):
            with torch.no_grad():
                pred_mod = self.model(start, **kwargs)
                if self.postprocess is not None:
                    pred_mod = self.postprocess(pred_mod)
            pred_mod = torch.softmax(pred_mod, dim=-1)
            scores[:,i] = pred_mod[range(bsz), c]
            # Render image if verbose, if it's the last step or if save is required.
            
            if i < n_steps:
                coords = salient_order[:, :, self.step * i:self.step * (i + 1)]
                batch_indices = torch.arange(bsz).view(-1, 1, 1).to(coords.device)

                channel_indices = torch.arange(n_channel).view(1, -1, 1).to(coords.device)
                start.reshape(bsz, n_channel, HW)[batch_indices, 
                                                  channel_indices, 
                                                  coords] = finish.reshape(bsz, n_channel, HW)[batch_indices, 
                                                                                               channel_indices, 
                                                                                               coords]
            all_states.append(start.clone())
            if (start == finish).all():
                for j in range(i+1, n_steps+1):
                    scores[:, j] = scores[:, j - 1]
                break
            
        auc_score = self.auc(scores)
        if return_dict:
            return {
                'auc_score': auc_score,
                'scores': scores,
                'start': start_clone,
                'finish': finish,
                'all_states': all_states
            }
        else:
            return auc_score
    
    def plot(self, img_tensor, scores, save_to=None):
        n_steps = scores.shape[-1] - 1
        i = n_steps
        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, 
                                                scores[i]))
        plt.axis('off')
        plt.imshow(img_tensor.cpu().numpy().transpose(1, 2, 0))

        plt.subplot(122)
        plt.plot(np.arange(i+1) / n_steps, scores[:i+1].cpu().numpy())
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(np.arange(i+1) / n_steps, 0, 
                         scores[:i+1].cpu().numpy(), 
                         alpha=0.4)
        plt.title(title)
        plt.xlabel(ylabel)
        # plt.ylabel(get_class_name(c))
        if save_to:
            plt.savefig(save_to + '/{}_{:03d}.png'.format(self.mode, i))
            plt.close()
        else:
            plt.show()

    @classmethod
    def gkern(cls, klen, nsig, num_channels):
        """Returns a Gaussian kernel array.
        Convolution with it results in image blurring."""
        
        # create nxn zeros
        inp = torch.zeros(klen, klen)
        # set element at the middle to one, a dirac delta
        inp[klen//2, klen//2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        k = gaussian_filter(inp, nsig)
        k = torch.tensor(k)
        kern = torch.zeros((num_channels, num_channels, klen, klen)).float()
        for i in range(num_channels):
            kern[i, i] = k
        return kern
    
    @classmethod
    def get_gaussian_kernel(cls, klen=11, nsig=5, channels=3):
        """Returns a Gaussian kernel tensor.
        Convolution with it results in image blurring."""
        kernel = cls.gkern(klen=klen, nsig=nsig, num_channels=channels)
        return kernel
    

class DeletionCls(InsDelCls):

    def __init__(self, model, step=224, substrate_fn=torch.zeros_like, 
                 postprocess=None):
        mode = 'del'
        super(DeletionCls, self).__init__(model, mode, step, substrate_fn, postprocess)


class InsertionCls(InsDelCls):

    def __init__(self, model, step=224, substrate_fn=torch.zeros_like, 
                 postprocess=None):
        mode = 'ins'
        super(InsertionCls, self).__init__(model, mode, step, substrate_fn, postprocess)


class InsDelSem(InsDelCls):

    def __init__(self, model, mode, step, substrate_fn, postprocess=None):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(InsDelSem, self).__init__(model, mode, step, substrate_fn, postprocess)

    def forward(self, X, Z, sem_part, kwargs={}, return_dict=False):
        """Run metric on one image-saliency pair.
            Args:
                X = img_tensor (Tensor): normalized image tensor. (bsz, n_channel, img_dim1, img_dim2)
                Z = explanation (Tensor): saliency map. (bsz, 1, img_dim1, img_dim2)
                sem_part: (bsz, 1, img_dim1, img_dim2)
                verbose (int): in [0, 1, 2].
                    0 - return list of scores.
                    1 - also plot final step.
                    2 - also plot every step and print 2 top classes.
                save_to (str): directory to save every step plots to.
            Return:
                scores (Tensor): Array containing scores at every step.
        """
        self.model.eval()
        auc_score_all = []
        scores_all = []
        starts = []
        finishes = []
        for b_i in range(X.size(0)):
            sem_part_bool = convert_idx_masks_to_bool(sem_part[b_i:b_i+1])
            num_masks = sem_part_bool.size(0)

            img_tensor = X[b_i:b_i+1]
            explanation = Z[b_i:b_i+1].to(img_tensor.device)

            bsz, n_channel, img_dim1, img_dim2 = img_tensor.shape
            HW = img_dim1 * img_dim2
            kwargs_i = {k: v[b_i:b_i+1] for k, v in kwargs.items()}
            with torch.no_grad():
                pred = self.model(img_tensor, **kwargs_i)
                if self.postprocess is not None:
                    pred = self.postprocess(pred)
            top, c = torch.max(pred, 1)

            if self.mode == 'del':
                start = img_tensor.clone()
                finish = self.substrate_fn(img_tensor)
            elif self.mode == 'ins':
                start = self.substrate_fn(img_tensor)
                finish = img_tensor.clone()

            start[start < 0] = 0.0
            start[start > 1] = 1.0
            finish[finish < 0] = 0.0
            finish[finish > 1] = 1.0

            t_r_masks = (explanation * sem_part_bool.unsqueeze(1).float()).reshape(num_masks, 
                                                                                -1).mean(-1)
            salient_order_masks = torch.argsort(t_r_masks, dim=-1).flip(-1)

            n_steps = len(salient_order_masks)

            scores = torch.empty(bsz, n_steps + 1).cuda()
            # Coordinates of pixels in order of decreasing saliency
            for i in range(n_steps+1):
                with torch.no_grad():
                    pred_mod = self.model(start, **kwargs_i)
                    if self.postprocess is not None:
                        pred_mod = self.postprocess(pred_mod)

                pred_mod = torch.softmax(pred, dim=-1)
                scores[:,i] = pred_mod[range(bsz), c]
                
                if i < n_steps:
                    mask_sem_best = sem_part_bool[salient_order_masks[i]]
                    start[0,:,mask_sem_best] = finish[0,:,mask_sem_best]

            auc_score = self.auc(scores)
            
            auc_score_all.append(auc_score)
            scores_all.append(scores)
            starts.append(start)
            finishes.append(finish)
        
        if return_dict:
            return {
                'auc_score': torch.stack(auc_score_all),
                'scores': scores_all,
                'start': torch.stack(starts),
                'finish': torch.stack(finishes)
            }
        else:
            return torch.stack(auc_score_all)
