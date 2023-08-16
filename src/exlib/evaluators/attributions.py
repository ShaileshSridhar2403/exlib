import torch
import torch.nn as nn

class Evaluator(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """
	def __init__(self, model): 
		super(Evaluator, self).__init__() 
		self.model = model

	def forward(self, X, Z): 
		""" Given a minibatch of examples X and feature attributions Z, 
		evaluate the quality of the feature attribution. """
		raise NotImplementedError()

class NNZ(Evaluator): 
	def __init__(self): 
		super(NNZ, self).__init__(None)

	def forward(self, X, Z, tol=1e-5): 
		n = Z.size(0)
		Z = (Z.abs() > tol)
		return torch.count_nonzero(Z.view(n,-1),dim=1)