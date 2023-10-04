import torch
import torch.nn.functional as F
import lime
import numpy as np
from lime import lime_image
from .common import FeatureAttrOutput, torch_img_to_np, np_to_torch_img


def lime_image_class_wrapper_fn(model, preprocess=None, postprocess=None):
    def go(X):
        model.eval()
        if preprocess is not None:
            X = preprocess(X)

        X = X.to(next(model.parameters()).device)
        y = model(X)
        if postprocess is not None:
            y = postprocess(y)

        return y.detach().cpu().numpy()

    return go


def lime_image_preprocess_np_to_pt(X_np, collapse):
    X = np_to_torch_img(X_np)
    if collapse:
        X = X[:,0:1,:,:] # even though lime needs no singleton last dimension in its input,
        # for an odd reason they put back 3 of them to match RGB format before passing
        # to batch_predict. So we need to remove the extraneous ones.
    return X


def explain_image_with_lime(X, model, label=None,
                            normalize_input=False,
                            postprocess=None,
                            LimeImageExplainerKwargs={},
                            # Gets FA for every label if top_labels == None
                            explain_instance_kwargs={},
                            get_image_and_mask_kwargs={}):
    """
    Explain a pytorch model with LIME.

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000
    batch_size=10, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=True

    # get_image_and_mask arguments
    positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """
    device = next(model.parameters()).device

    X_min, X_max = X.min(), X.max()
    if normalize_input:
        X = (X - X_min)/(X_max-X_min) # shift to 0-1 range
    X_np = torch_img_to_np(X.cpu()) # rearrange dimensions for numpy

    collapse = (X.ndim == 4) and (X.size(1) == 1) # check if single or RGB channel
    if collapse:
        X_np = X_np[:,:,:,0] # lime needs no singleton last dimension

    preprocess = lambda X_np : lime_image_preprocess_np_to_pt(X_np, collapse)

    f = lime_image_class_wrapper_fn(model, preprocess=preprocess, postprocess=postprocess)

    # Uniformly conver the label if appropriate
    label = label if (label is None or isinstance(label, torch.Tensor)) else torch.tensor(label)
    masks, lime_exps = [], []
    for i, Xi_np in enumerate(X_np):
        explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)
        # Need to do some tricks to not have Lime give a key-value error ... yet default is (1,)?
        todo_label = 1 if label is None else label[i].cpu().item()
        lime_exp = explainer.explain_instance(Xi_np, f, labels=(todo_label,), **explain_instance_kwargs)
        todo_label = lime_exp.top_labels[0] if label is None else label[i].cpu().item()
        img, mask = lime_exp.get_image_and_mask(todo_label, **get_image_and_mask_kwargs)
        masks.append(mask)
        lime_exps.append(lime_exp)

    return FeatureAttrOutput(torch.from_numpy(np.stack(masks)), lime_exps)


################################################################################
### Old stuff below to be deleted later:

def batch_predict_from_torch(model, task, preprocess=None, postprocess=None): 
    """ Batch predict function for a pytorch model """
    def batch_predict(inp):
        model.eval()
        if preprocess is not None: 
            inp = preprocess(inp) 

        inp = inp.to(next(model.parameters()).device)

        pred = model(inp)
        if postprocess is not None:
            pred = postprocess(pred)
        if task == 'reg': 
            output = pred
        elif task == 'clf': 
            output = F.softmax(pred, dim=1)
        else: 
            raise ValueError(f"Task {task} not implemented")
        return output.detach().cpu().numpy()
#         assert False
    return batch_predict

def explain_torch_reg_with_lime(X, model, label, postprocess=None,
                                normalize=False, LimeImageExplainerKwargs={}, 
                                explain_instance_kwargs={}, 
                                get_image_and_mask_kwargs={}): 
    """
    Explain a pytorch model with LIME. 

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000 
    batch_size=10, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=True

    # get_image_and_mask arguments
    positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """
    device = next(model.parameters()).device
    collapse = (X.ndim == 4) and (X.size(1) == 1) # check if single or RGB channel
    X_min, X_max = X.min(), X.max()
    if normalize: 
        X = (X - X_min)/(X_max-X_min) # shift to 0-1 range
    X_np = torch_img_to_np(X.cpu()) # rearrange dimensions for numpy
    if collapse: 
        X_np = X_np[:,:,:,0] # lime needs no singleton last dimension
        
    def p(X): 
        X = np_to_torch_img(X).to(device)
        if collapse: 
            X = X[:,0:1,:,:] # even though lime needs no singleton last dimension in its input, 
            # for an odd reason they put back 3 of them to match RGB format before passing 
            # to batch_predict. So we need to remove the extraneous ones. 

        if normalize: 
            X = X*(X_max - X_min) + X_min # undo shift
        return X
        
    f = batch_predict_from_torch(model, 'reg', preprocess=p, 
                                 postprocess=postprocess)
    
    masks,lime_exps = [],[]
    for i, X0_np in enumerate(X_np): 
        explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)
        explanation = explainer.explain_instance(X0_np, f, **explain_instance_kwargs)
        # print('label', label)
        # print('explanation.top_labels[0]', explanation.top_labels[0])
        # import pdb
        # pdb.set_trace()
        # print('explanation', explanation)

        if label is None:
            todo_label = explanation.top_labels[0]
        elif isinstance(label, torch.Tensor):
            todo_label = label[i].cpu().item()
        else:
            todo_label = label[i]

        img,mask = explanation.get_image_and_mask(todo_label, **get_image_and_mask_kwargs)

        masks.append(mask)
        lime_exps.append(explanation)

    return FeatureAttrOutput(torch.from_numpy(np.stack(masks)), lime_exps)

