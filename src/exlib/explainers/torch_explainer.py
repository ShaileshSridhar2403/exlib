import torch.nn as nn
from .lime import explain_torch_reg_with_lime

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
	def __init__(self, model, normalize=False, **kwargs):
		super(TorchImageLime, self).__init__(model) 
		self.normalize = normalize
		self.kwargs = kwargs

	def forward(self, X, label=None): 
		return explain_torch_reg_with_lime(X, self.model, label, 
			normalize=self.normalize, **self.kwargs)