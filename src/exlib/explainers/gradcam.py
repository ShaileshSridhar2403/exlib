# TODO: fix gradcam itself to make it generalize

import torch
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from .common import *
from copy import deepcopy


class WrappedModel(torch.nn.Module):
    def __init__(self, model): 
        super(WrappedModel, self).__init__()
        self.model = model
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            
    def forward(self, x):
        return self.model(x)


class GradCAMImageCls(FeatureAttrMethod):
    def __init__(self, model, target_layers, reshape_transform=None):
        
        model = WrappedModel(model)

        super().__init__(model)
        
        self.target_layers = target_layers
        with torch.enable_grad():
            self.grad_cam = GradCAM(model=model, target_layers=self.target_layers,
                                    reshape_transform=reshape_transform,
                                    use_cuda=True if torch.cuda.is_available() else False)

    def forward(self, X, label=None, target_func=ClassifierOutputSoftmaxTarget):
        with torch.enable_grad():
            grad_cam_result = self.grad_cam(input_tensor=X, targets=[target_func(label) for label in label])
            grad_cam_result = torch.tensor(grad_cam_result)

        return FeatureAttrOutput(grad_cam_result.unsqueeze(1), grad_cam_result)
