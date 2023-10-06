import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import shap

from .common import FeatureAttrMethod
from .shap import explain_torch_with_shap


# The default behavior for an attribution method is to
# provide an explanation for the top predicted class.
# Initialize defaults in the init function.


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



