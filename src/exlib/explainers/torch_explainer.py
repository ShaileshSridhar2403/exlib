from lime import explain_torch_reg_with_lime

class TorchAttribution(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """

	def forward(self, X): 
		""" Given a minibatch of examples X, generate a feature 
		attribution for each example. """
		raise NotImplementedError()

class TorchLime(nn.Module): 
	def __init__(self, model, normalize=False, **kwargs):
		super(TorchLime, self).__init__() 
		self.model = model
		self.normalize = normalize
		self.kwargs = kwargs

	def forward(self, X): 
		return explain_torch_reg_with_lime(X, self.model, 
			normalize=self.normalize, **self.kwargs)