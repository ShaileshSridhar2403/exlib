import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import shap

from .common import TorchAttribution
from .lime import explain_torch_reg_with_lime
from .shap import explain_torch_with_shap
from .rise import TorchImageRISE
from .intgrad import explain_torch_with_intgrad

# The default behavior for an attribution method is to 
# provide an explanation for the top predicted class. 
# Initialize defaults in the init function. 


class TorchImageLime(TorchAttribution): 
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

class TorchImageSHAP(TorchAttribution): 
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
        #max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if labels is None else labels)

class TorchImageIntGrad(TorchAttribution):
    def __init__(self, model, postprocess=None):
        super(TorchImageIntGrad, self).__init__(model, postprocess)

    def forward(self, X, labels=None, **kwargs):
        if labels is None:
            y = self.model(X)
            if self.postprocess:
                y = self.postprocess(y)
            labels = y.argmax(dim=1)

        def y_to_loss(y):
            assert y.ndim == 2
            assert len(labels) == y.size(0)
            loss = 0.0
            for i, l in enumerate(labels):
                loss += y[i,l]
            return loss

        return explain_torch_with_intgrad(X, self.model, y_to_loss, 
                                          postprocess=self.postprocess,
                                          **kwargs)

class TorchImageSegIntGrad(TorchAttribution):
    def __init__(self, model, postprocess=None):
        super(TorchImageSegIntGrad, self).__init__(model, postprocess)

    def forward(self, X, labels, **kwargs):

        def y_to_loss(y):
            assert len(labels) == y.size(0)
            assert X.shape[2:] == y.shape[2:] # The non-batch and non-channel dims
            loss = 0.0
            for i, l in enumerate(labels):
                inds = y.argmax(dim=1) # Max along the channels
                H = F.one_hot(inds, num_classes=y.size(1))
                H = H.unsqueeze(1).transpose(1,-1).view(y.shape)
                L = (y * H).flatten(1).sum(dim=1)
                return L

        return explain_torch_with_intgrad(X, self.model, y_to_loss, 
                                          postprocess=self.postprocess,
                                          **kwargs)

