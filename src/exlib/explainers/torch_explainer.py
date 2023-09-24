import torch.nn as nn
import copy

import shap

from .common import TorchAttribution
from .lime import explain_torch_reg_with_lime, explain_torch_reg_with_lime_text
from .shap import explain_torch_with_shap, explain_torch_with_shap_text
from .rise import TorchImageRISE
from .intgrad import explain_torch_with_intgrad, explain_torch_with_intgrad_text
from lime import lime_image, lime_text

# The default behavior for an attribution method is to 
# provide an explanation for the top predicted class. 
# Initialize defaults in the init function. 


class TorchImageLime(TorchAttribution): 
    def __init__(self, model, postprocess=None,
                   normalize=False, 
                 LimeImageExplainerKwargs={}, 
                 explain_instance_kwargs={}, 
                 get_image_and_mask_kwargs={},
                 task='reg'):
        super().__init__(model, postprocess) 
        self.normalize = normalize
        self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs
        self.task = task

    def forward(self, X, label=None): 
        return explain_torch_reg_with_lime(X, self.model, label, 
            postprocess=self.postprocess,
            normalize=self.normalize, 
            LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
            explain_instance_kwargs=self.explain_instance_kwargs,
            get_image_and_mask_kwargs=self.get_image_and_mask_kwargs,
            task=self.task)


class TorchTextLime(TorchAttribution): 
    def __init__(self, model, tokenizer, postprocess=None,
                   normalize=False, 
                 LimeTextExplainerKwargs={}, 
                 explain_instance_kwargs={}, 
                 get_image_and_mask_kwargs={},
                 task='reg',
                 batch_size=16):
        super().__init__(model, postprocess) 
        self.tokenizer = tokenizer
        self.normalize = normalize
        self.LimeTextExplainerKwargs = LimeTextExplainerKwargs
        self.explain_instance_kwargs = explain_instance_kwargs
        self.get_image_and_mask_kwargs = get_image_and_mask_kwargs
        self.task = task
        self.batch_size = batch_size
        self.explainer = lime_text.LimeTextExplainer(**LimeTextExplainerKwargs)

    def forward(self, X, label=None): 
        return explain_torch_reg_with_lime_text(X, self.model, 
                                                self.tokenizer, label, 
            postprocess=self.postprocess,
            normalize=self.normalize, 
            # LimeTextExplainerKwargs=self.LimeTextExplainerKwargs,
            explainer=self.explainer,
            explain_instance_kwargs=self.explain_instance_kwargs,
            get_image_and_mask_kwargs=self.get_image_and_mask_kwargs,
            task=self.task,
            batch_size=self.batch_size)


class TorchImageSHAP(TorchAttribution): 
    def __init__(self, model, postprocess=None,
                 mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super().__init__(model, postprocess) 

        # default to just explaining the top class
        if "outputs" not in shap_kwargs: 
            shap_kwargs["outputs"] = shap.Explanation.argsort.flip[:1]

        self.mask_value = mask_value
        self.explainer_kwargs = explainer_kwargs
        self.shap_kwargs = shap_kwargs

    def forward(self, X, label=None): 
        sk = self.shap_kwargs
        if label is not None:         
            # import pdb
            # pdb.set_trace()
            # sk = {k: copy.deepcopy(v) for k, v in self.shap_kwargs.items()}
            try:
                sk = copy.deepcopy(self.shap_kwargs)
            except:
                sk = {k: v for k, v in self.shap_kwargs.items() if k != 'outputs'}
            sk["outputs"] = label

        return explain_torch_with_shap(X, self.model, self.mask_value, 
            self.explainer_kwargs, self.shap_kwargs, postprocess=self.postprocess)
        #max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if labels is None else labels)


class TorchTextSHAP(TorchAttribution): 
    def __init__(self, model, tokenizer, postprocess=None,
                 mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super().__init__(model, postprocess) 

        # default to just explaining the top class
        if "outputs" not in shap_kwargs: 
            shap_kwargs["outputs"] = shap.Explanation.argsort.flip[:1]

        self.tokenizer = tokenizer
        self.mask_value = mask_value
        self.explainer_kwargs = explainer_kwargs
        self.shap_kwargs = shap_kwargs

    def forward(self, X, label=None): 
        sk = self.shap_kwargs
        if label is not None:         
            sk = copy.deepcopy(self.shap_kwargs)
            sk["outputs"] = label

        return explain_torch_with_shap_text(X, self.model, self.tokenizer, self.mask_value, 
            self.explainer_kwargs, self.shap_kwargs, postprocess=self.postprocess)
        #max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if labels is None else labels)


class TorchImageIntGrad(TorchAttribution):
    def __init__(self, model, postprocess=None):
        super().__init__(model, postprocess)

    def forward(self, X, label=None):
        return explain_torch_with_intgrad(X, self.model, labels=label, 
                                          postprocess=self.postprocess)
    

class TorchTextIntGrad(TorchAttribution):
    def __init__(self, model, postprocess=None, mask_combine=None):
        super().__init__(model, postprocess)
        self.mask_combine = mask_combine

    def forward(self, X, label=None, kwargs={}):
        return explain_torch_with_intgrad_text(X, self.model, labels=label, 
                                          postprocess=self.postprocess, kwargs=kwargs, 
                                          mask_combine=self.mask_combine)



