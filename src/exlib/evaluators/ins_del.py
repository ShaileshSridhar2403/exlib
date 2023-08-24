import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .common import Evaluator

class InsDel(Evaluator):

    def __init__(self, model, mode, step, substrate_fn, postprocess=None):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(InsDel, self).__init__(model, postprocess)
        
        assert mode in ['del', 'ins']
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn

    def auc(self, arr):
        """Returns normalized Area Under Curve of the array."""
        # return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
        # return (arr.sum(-1).sum(-1) - arr[1] / 2 - arr[-1] / 2) / (arr.shape[1] - 1)
        return (arr.sum(-1) - arr[:, 0] / 2 - arr[:, -1] / 2) / (arr.shape[1] - 1)

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
        top, c = torch.max(pred, 1)
        # c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        start[start < 0] = 0.0
        start[start > 1] = 1.0
        finish[finish < 0] = 0.0
        finish[finish > 1] = 1.0

        scores = torch.empty(bsz, n_steps + 1).cuda()
        # Coordinates of pixels in order of decreasing saliency
        t_r = explanation.reshape(bsz, -1, HW)
        salient_order = torch.argsort(t_r, dim=-1)
        salient_order = torch.flip(salient_order, [1, 2])
        for i in range(n_steps+1):
            pred = self.model(start)
            if self.postprocess is not None:
                pred = self.postprocess(pred)
            pred = torch.softmax(pred, dim=-1)
            # pr, cl = torch.topk(pred, 2)
            # if verbose == 2:
            #     print('{}: {:.3f}'.format(get_class_name(cl[0][0]), float(pr[0][0])))
            #     print('{}: {:.3f}'.format(get_class_name(cl[0][1]), float(pr[0][1])))
            scores[:,i] = pred[range(bsz), c]
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
        
        auc_score = self.auc(scores)
        if return_dict:
            return {
                'auc_score': auc_score,
                'scores': scores,
                'start': start,
                'finish': finish
            }
        else:
            return auc_score
    
    def plot(self, n_steps, start, scores, save_to=None):
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
        plt.imshow(start[0].cpu().numpy().transpose(1, 2, 0))

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
    def gkern(cls, klen, nsig):
        """Returns a Gaussian kernel array.
        Convolution with it results in image blurring."""
        
        # create nxn zeros
        inp = torch.zeros(klen, klen)
        # set element at the middle to one, a dirac delta
        inp[klen//2, klen//2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        k = gaussian_filter(inp, nsig)
        k = torch.tensor(k)
        kern = torch.zeros((3, 3, klen, klen)).float()
        kern[0, 0] = k
        kern[1, 1] = k
        kern[2, 2] = k
        return kern