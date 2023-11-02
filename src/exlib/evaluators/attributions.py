import torch
import torch.nn as nn
from .common import Evaluator
from .ins_del import InsDelCls, InsDelSem, DeletionCls, InsertionCls
from .comp_suff import CompSuff, CompSuffSem

class NNZ(Evaluator): 
    def __init__(self): 
        super(NNZ, self).__init__(None)

    def forward(self, X, Z, tol=1e-5, normalize=False): 
        n = Z.size(0)
        Z = (Z.abs() > tol).reshape(n,-1)
        nnz = torch.count_nonzero(Z,dim=1)
        if normalize: 
            return nnz / Z.size(1)
        else: 
            return nnz
        

class NNZGroup(Evaluator): 
    def __init__(self, model, postprocess): 
        super(NNZGroup, self).__init__(model, postprocess)

    def forward(self, X, Z, kwargs=None, W=None, pred=None, tol=1e-5, normalize=False): 
        """
        X: (n, **l)
        Z: (n, m, **l)
        W: (n, m, c)
        """
        n = Z.size(0)
        m = Z.size(1)
        if W is None:
            W = torch.ones(n, m)
        else:
            if pred is None:
                if kwargs is None:
                    pred = self.model(X)
                else:
                    pred = self.model(X, **kwargs)
                if self.postprocess:
                    pred = self.postprocess(pred)
                # pred (n, c)
                _, pred = torch.max(pred, 1)
            W = W[range(n),:,pred]
        
        Z = (Z.abs() > tol).reshape(n,m,-1)
        nnz = torch.count_nonzero(Z,dim=-1)
        # import pdb
        # pdb.set_trace()
        if normalize: 
            return (nnz * W).sum(-1) / W.sum(-1) / Z.size(-1)
        else: 
            return (nnz * W).sum(-1) / W.sum(-1)


class Consistency(Evaluator): 
    def __init__(self, model, postprocess, task='cls'): 
        super(Consistency, self).__init__(model, postprocess)
        assert (task in ['cls', 'reg', 'multicls'])
        self.task = task

    def forward(self, X, Z, kwargs=None, pred=None): 
        import pdb
        pdb.set_trace()
        print(X.shape)
        print(Z.shape)
        print((X * Z.bool()).shape)
        with torch.no_grad():
            if X.shape != Z.shape:
                Z = Z.squeeze(1)
            if pred is None:
                if kwargs is None:
                    pred = self.model(X)
                else:
                    pred = self.model(X, **kwargs)
                if self.postprocess:
                    pred = self.postprocess(pred)
                if self.task == 'cls':
                    _, pred = torch.max(pred, 1)
                elif self.task == 'multicls':
                    pred = torch.sigmoid(pred)
                    pred = (pred > 0.5).float()

            
            if kwargs is None:
                pred_mod = self.model(X * Z.bool())
            else:
                pred_mod = self.model(X * Z.bool(), **kwargs)
            if self.postprocess:
                pred_mod = self.postprocess(pred_mod)
            if self.task == 'cls':
                _, pred_mod = torch.max(pred_mod, 1)
                result = (pred == pred_mod).sum()
            elif self.task == 'multicls':
                pred_mod = torch.sigmoid(pred_mod)
                pred_mod = (pred_mod > 0.5).float()
                result = (pred == pred_mod).sum()
            else: # reg
                criterion = nn.MSELoss()
                result = criterion(pred, pred_mod) * pred.size(0)
            return result