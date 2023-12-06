import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .common import *
from .libs.archipelago.explainer import Archipelago
from .libs.archipelago.application_utils.image_utils import *
from .libs.archipelago.application_utils.utils_torch import ModelWrapperTorch
import warnings
warnings.filterwarnings("ignore")


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


class ArchipelagoImageCls(FeatureAttrMethod):
    """ Image classification with integrated gradients
    """
    def __init__(self, model, top_k=5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_wrapper = ModelWrapperTorch(model, device)
        super().__init__(model_wrapper)
        self.top_k = top_k

    def forward(self, x, t, **kwargs):
        bsz = x.shape[0]

        if not isinstance(t, torch.Tensor) and t is not None:
            t = torch.tensor(t)

        expln_scores_all = []
        expln_flat_masks_all = []
        masks_all = []
        mask_weights_all = []
        for i in range(bsz):
            image = x[i].cpu().permute(1,2,0).numpy()
            if t is None:
                predictions = self.model(np.expand_dims(image,0))
                class_idx = predictions[0].argsort()[::-1][0]
            else:
                class_idx = t[i].cpu().item()

            baseline = np.zeros_like(image)
            segments = quickshift(image, kernel_size=3, max_dist=300, ratio=0.2)

            xf = ImageXformer(image, baseline, segments)
            apgo = Archipelago(self.model, data_xformer=xf, output_indices=class_idx, batch_size=20)
            explanation = apgo.explain(top_k=self.top_k)

            expln_scores = np.zeros_like(segments, dtype=float)
            expln_flat_masks = np.zeros_like(segments, dtype=float)
            masks = []
            mask_weights = []


            for e_i, (k, v) in enumerate(sorted(explanation.items(), key=lambda item: item[1], reverse=True)):
                mask = np.zeros_like(segments, dtype=float)
                for s_i in k:
                    expln_scores[segments == s_i] = v
                    expln_flat_masks[segments == s_i] = e_i
                    mask[segments == s_i] = 1
                masks.append(mask)
                mask_weights.append(v)

            expln_scores = torch.tensor(expln_scores).unsqueeze(0).to(x.device)
            expln_flat_masks = torch.tensor(expln_flat_masks).unsqueeze(0).to(x.device)
            masks = torch.tensor(masks).to(x.device)
            mask_weights = torch.tensor(mask_weights).to(x.device)
            expln_scores_all.append(expln_scores)
            expln_flat_masks_all.append(expln_flat_masks)
            masks_all.append(masks)
            mask_weights_all.append(mask_weights)
        
        expln_scores = torch.stack(expln_scores_all, dim=0)
        expln_flat_masks = torch.stack(expln_flat_masks_all, dim=0)

        return FeatureAttrOutput(expln_scores, {
            "expln_flat_masks": expln_flat_masks,
            "masks": masks_all,
            "mask_weights": mask_weights_all,
        })
