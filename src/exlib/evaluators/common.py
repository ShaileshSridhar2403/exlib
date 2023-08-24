import torch
import torch.nn as nn

class Evaluator(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """
	def __init__(self, model, postprocess=None): 
		super(Evaluator, self).__init__() 
		self.model = model
		self.postprocess = postprocess

	def forward(self, X, Z): 
		""" Given a minibatch of examples X and feature attributions Z, 
		evaluate the quality of the feature attribution. """
		raise NotImplementedError()