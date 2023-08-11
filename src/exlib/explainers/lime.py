import torch
import torch.nn.functional as F
import lime
from lime import lime_image

def patch_segmenter(image, sz=(8,8)): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    X = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=X.size()[:2], mode='nearest').long()
    segments = segments[0,0].numpy()
    return segments

def batch_predict_from_torch(model, task, preprocess=None): 
    """ Batch predict function for a pytorch model """
    def batch_predict(inp):
        model.eval()
        if preprocess is not None: 
            inp = preprocess(inp)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inp = inp.to(device)

        pred = model(inp)
        if task == 'reg': 
            output = pred
        elif task == 'clf': 
            output = F.softmax(pred, dim=1)
        else: 
            raise ValueError(f"Task {task} not implemented")
        return output.detach().cpu().numpy()
#         assert False
    return batch_predict

def explain_torch_reg_with_lime(X, model, normalize=False, **kwargs): 
    """ Explain a pytorch model with LIME. Use patches as default. """
    collapse = (X.ndim == 4) and (X.size(1) == 1)
    X_min, X_max = X.min(), X.max()
    if normalize: 
        X = (X - X_min)/(X_max-X_min) # shift to 0-1 range
    X_np = X.permute(0,2,3,1).numpy() # rearrange dimensions for numpy
    if collapse: 
        X_np = X_np[:,:,:,0] # lime needs no singleton last dimension
        
    def p(X): 
        X = torch.from_numpy(X)
        if collapse: 
            X = X[:,:,:,0:1] # even though limt needs no singleton last dimension in its input, 
            # for an odd reason they put back 3 of them to match RGB format before passing 
            # to batch_predict. So we need to remove the extraneous ones. 
        X = X.permute(0,3,1,2) # undo permutation
        if normalize: 
            X = X*(X_max - X_min) + X_min # undo shift
        return X
        
    f = batch_predict_from_torch(model, 'reg', preprocess=p)
    
    lime_exps = []
    for X0_np in X_np: 
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(X0_np, f, **kwargs)
        lime_exps.append(explanation)

    return {
        "lime_explanations" : lime_exps
    }
    
