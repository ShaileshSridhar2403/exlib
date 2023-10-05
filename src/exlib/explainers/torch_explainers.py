import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import shap

from .common import FeatureAttrMethod
from .lime import explain_torch_reg_with_lime, explain_image_with_lime
from .shap import explain_torch_with_shap
from .rise import TorchImageRISE

# The default behavior for an attribution method is to
# provide an explanation for the top predicted class.
# Initialize defaults in the init function.

class LimeImageCls(FeatureAttrMethod):
    def __init__(self, model, postprocess=None, normalize_input=False,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # This is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageCls, self).__init__(model, postprocess)
        self.normalize_input = normalize_input
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

    def forward(self, X, label=None):
        return explain_image_with_lime(X, self.model, label,
            postprocess=self.postprocess,
            normalize_input=self.normalize_input,
            LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
            explain_instance_kwargs=self.explain_instance_kwargs,
            get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)


# Segmentation model
class LimeImageSeg(FeatureAttrMethod):
    def __init__(self, model, postprocess=None, normalize_input=False,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={
                     # Make this however big you need to get every label
                     # This is because the original LIME API is stupid
                     "top_labels" : 1000000,
                     "num_samples" : 500,
                 },
                 get_image_and_mask_kwargs={}):
        super(LimeImageSeg, self).__init__(model, postprocess)
        self.normalize_input = normalize_input
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

    def forward(self, X, label=None):

        # LIME fundamentally works with image classification models,
        # so we wrap the output of the segmentation model to mimic classification
        def seg_postprocess(y):
            y = self.postprocess(y)
            N, K, H, W = y.shape
            A = y.argmax(dim=1, keepdim=True)
            M = [A == k for k in range(K)]
            L = [(y * M[k]).sum(dim=(1,2,3)).view(N,1) for k in range(K)]
            return torch.cat(L, dim=1)

        return explain_image_with_lime(X, self.model, label,
            postprocess=seg_postprocess,
            normalize_input=self.normalize_input,
            LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
            explain_instance_kwargs=self.explain_instance_kwargs,
            get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)


# Old implementation of a LIME wrapper
class TorchImageLime(FeatureAttrMethod):
    def __init__(self, model, postprocess=None,
                   normalize=False,
                 LimeImageExplainerKwargs={},
                 explain_instance_kwargs={},
                 get_image_and_mask_kwargs={}):
        super(TorchImageLime, self).__init__(model, postprocess)
        self.normalize = normalize
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

    def forward(self, X, label=None):
        return explain_torch_reg_with_lime(X, self.model, label,
            postprocess=self.postprocess,
            normalize=self.normalize,
            LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
            explain_instance_kwargs=self.explain_instance_kwargs,
            get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)


class ShapImageCls(FeatureAttrMethod):
    def __init__(self, model, postprocess=None,
                 mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super(ShapImageCls, self).__init__(model, postprocess)


# Old implementation of a SHAP wrapper
class TorchImageSHAP(FeatureAttrMethod):
    def __init__(self, model, postprocess=None,
                 mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super(TorchImageSHAP, self).__init__(model, postprocess)

        # default to just explaining the top class
        if "outputs" not in shap_kwargs:
            shap_kwargs["outputs"] = shap.Explanation.argsort.flip[:1]

        self.mask_value = mask_value
        self.explainer_kwargs = explainer_kwargs
        self.shap_kwargs = shap_kwargs

    def forward(self, X, label=None):
        sk = self.shap_kwargs
        if label is not None:
            sk = copy.deepcopy(self.shap_kwargs)
            sk["outputs"] = label

        return explain_torch_with_shap(X, self.model, self.mask_value,
            self.explainer_kwargs, self.shap_kwargs, postprocess=self.postprocess)
        #max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if label is None else label)



