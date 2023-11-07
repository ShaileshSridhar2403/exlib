import copy
import torch
import torch.nn.functional as F
import shap
from .common import *


def explain_image_cls_with_shap(model, x, t, mask_value, explainer_kwargs, shap_kwargs): 
    device = next(model.parameters()).device
    x_np = torch_img_to_np(x.cpu())
    masker = shap.maskers.Image(mask_value, x_np[0].shape)

    def f(x): 
        with torch.no_grad(): 
            pred = model(np_to_torch_img(x).to(device))
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(f, masker, **explainer_kwargs)

    # here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_out = explainer(x_np, **shap_kwargs)
    shap_values = torch.from_numpy(shap_out.values).permute(0,3,1,2,4) # (N,H,W,C,num_classes)
    svs = []
    for i in range(x.size(0)):
        svs.append(shap_values[i,:,:,:,t[i]])
    svs = torch.stack(svs)
    return FeatureAttrOutput(svs, shap_out)


class ShapImageCls(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.explainer_kwargs = explainer_kwargs
        self.shap_kwargs = shap_kwargs


    def forward(self, x, t, **kwargs):
        sk = copy.deepcopy(self.shap_kwargs)
        # sk["outputs"] = t # Anton: I'm not 100% sure what this does, it _may_ help ... or not

        return explain_image_cls_with_shap(self.model, x, t, self.mask_value, self.explainer_kwargs, sk)


class ShapImageSeg(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, explainer_kwargs={}, shap_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.explainer_kwargs = explainer_kwargs
        self.shap_kwargs = shap_kwargs

        self.cls_model = Seg2ClsWrapper(model)


    def forward(self, x, t, **kwargs):
        sk = copy.deepcopy(self.shap_kwargs)
        sk["outputs"] = t

        return explain_image_cls_with_shap(self.cls_model, x, t, self.mask_value, self.explainer_kwargs, sk)

