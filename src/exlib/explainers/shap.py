import copy
import torch
import torch.nn.functional as F
import shap
import numpy as np
from .common import *



def explain_image_cls_with_shap(model, x, t, mask_value, shap_explainer_kwargs):
    assert len(x) == len(t)
    device = next(model.parameters()).device

    def f(x_np):
        with torch.no_grad():
            pred = model(np_to_torch_img(x_np).to(device))
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all partition explainer
    x_np = torch_img_to_np(x.cpu())
    masker = shap.maskers.Image(mask_value, x_np[0].shape)
    explainer = shap.Explainer(f, masker, **shap_explainer_kwargs)

    shap_outs = []
    shap_values = []
    for xi, ti in zip(x_np, t):
        out = explainer(np.expand_dims(xi, axis=0), outputs=[ti])
        svs = torch.from_numpy(out.values) # (1,H,W,C,1)
        shap_outs.append(out)
        shap_values.append(svs[0,:,:,:,0].permute(2,0,1)) # (C,H,W)

    shap_values = torch.stack(shap_values)
    return FeatureAttrOutput(shap_values, shap_outs)


class ShapImageCls(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, shap_explainer_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.shap_explainer_kwargs = shap_explainer_kwargs

    def forward(self, x, t, **kwargs):
        return explain_image_cls_with_shap(self.model, x, t, self.mask_value, self.shap_explainer_kwargs)


class ShapImageSeg(FeatureAttrMethod):
    def __init__(self, model, mask_value=0, shap_explainer_kwargs={}):
        super().__init__(model)
        self.mask_value = mask_value
        self.shap_explainer_kwargs = shap_explainer_kwargs
        self.cls_model = Seg2ClsWrapper(model)

    def forward(self, x, t, **kwargs):
        return explain_image_cls_with_shap(self.cls_model, x, t, self.mask_value, self.shap_explainer_kwargs)

