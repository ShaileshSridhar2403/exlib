import torch.nn as nn
import copy

import shap

from .lime import explain_torch_reg_with_lime
from .shap import explain_torch_with_shap

# The default behavior for an attribution method is to 
# provide an explanation for the top predicted class. 
# Initialize defaults in the init function. 

class TorchAttribution(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """
	def __init__(self, model): 
		super(TorchAttribution, self).__init__() 
		self.model = model

	def forward(self, X, label=None): 
		""" Given a minibatch of examples X, generate a feature 
		attribution for each example. If label is not specified, 
		explain the largest output. """
		raise NotImplementedError()

class TorchImageLime(TorchAttribution): 
	def __init__(self, model, normalize=False, 
				 LimeImageExplainerKwargs={}, 
                 explain_instance_kwargs={}, 
                 get_image_and_mask_kwargs={}):
		super(TorchImageLime, self).__init__(model) 
		self.normalize = normalize
		self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
		self.explain_instance_kwargs = explain_instance_kwargs
		self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

	def forward(self, X, label=None): 
		return explain_torch_reg_with_lime(X, self.model, label, 
			normalize=self.normalize, 
			LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
			explain_instance_kwargs=self.explain_instance_kwargs,
			get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)

class TorchImageSHAP(TorchAttribution): 
	def __init__(self, model, mask_value=0, explainer_kwargs={}, shap_kwargs={}):
		super(TorchImageSHAP, self).__init__(model) 

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
			self.explainer_kwargs, self.shap_kwargs)
		#max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if labels is None else labels)