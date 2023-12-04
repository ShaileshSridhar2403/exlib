# TODO: fix gradcam itself to make it generalize

import torch
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from .common import *
from copy import deepcopy


class WrappedModel(torch.nn.Module):
    def __init__(self, model, postprocess): 
        super(WrappedModel, self).__init__()
        self.model = model
        self.postprocess = postprocess
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            
    def forward(self, x):
        try:
            outputs = self.model(x, output_hidden_states=True)
        except:
            outputs = self.model(x)
        return outputs.logits


class GradCAMImageCls(FeatureAttrMethod):
    def __init__(self, model, target_layers, postprocess=None, reshape_transform=None):
        
        model = WrappedModel(model, postprocess)

        super().__init__(model, postprocess)
        
        self.target_layers = target_layers
        with torch.enable_grad():
            self.grad_cam = GradCAM(model=model, target_layers=self.target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=True if torch.cuda.is_available() else False)

    def forward(self, X, label=None):
        with torch.enable_grad():
            grad_cam_result = self.grad_cam(input_tensor=X, targets=label)
            grad_cam_result = torch.tensor(grad_cam_result)

        return FeatureAttrOutput(grad_cam_result.unsqueeze(1), grad_cam_result)
