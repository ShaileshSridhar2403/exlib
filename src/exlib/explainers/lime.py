import torch
import torch.nn.functional as F
import lime
import math
import numpy as np
from lime import lime_image, lime_text
from .common import AttributionOutput, torch_img_to_np, np_to_torch_img
from tqdm import tqdm

def batch_predict_from_torch(model, task, preprocess=None, postprocess=None, label=None): 
    """ Batch predict function for a pytorch model """
    def batch_predict(inp):
        model.eval()
        if preprocess is not None: 
            inp = preprocess(inp) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inp = inp.to(device)

        with torch.no_grad():
            pred = model(inp)
            if postprocess is not None:
                pred = postprocess(pred)
        if task == 'reg': 
            output = pred
        elif task == 'clf': 
            output = F.softmax(pred, dim=1)
        elif task == 'multiclf':
            output = F.sigmoid(pred)
            output = (output > 0.5).float()
            if label is not None:
                explain_labels = torch.abs(label) - 1
                label_pos = (label > 0)
                import pdb
                pdb.set_trace()
                print('output', output.shape)
                output = output[range(len(output)), label_pos]
                print('output', output.shape)
        else: 
            raise ValueError(f"Task {task} not implemented")
        return output.detach().cpu().numpy()
#         assert False
    return batch_predict

def explain_torch_reg_with_lime(X, model, label, postprocess=None,
                                normalize=False, LimeImageExplainerKwargs={}, 
                                explain_instance_kwargs={}, 
                                get_image_and_mask_kwargs={},
                                task='reg'): 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
    f = batch_predict_from_torch(model, task, preprocess=p, 
                                 postprocess=postprocess, label=label)
    
    masks,lime_exps = [],[]
    
    # [todo] change for future
    if task == 'multiclf' and label is not None:
        label = torch.abs(label) - 1
    
    for i, X0_np in tqdm(enumerate(X_np)): 
        explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)
        explanation = explainer.explain_instance(X0_np, f, **explain_instance_kwargs)
        # print('label', label)
        # print('explanation.top_labels[0]', explanation.top_labels[0])
        # import pdb
        # pdb.set_trace()
        # print('explanation', explanation)
        
        try:
            img,mask = explanation.get_image_and_mask(explanation.top_labels[0] 
                                                    if label is None else label[i].cpu().numpy().item(), 
                                                    **get_image_and_mask_kwargs)
        except:
            img,mask = explanation.get_image_and_mask(explanation.top_labels[0] 
                                                    if label is None else label[i].cpu().numpy().item())

        masks.append(mask)
        lime_exps.append(explanation)

    return AttributionOutput(torch.from_numpy(np.stack(masks)), lime_exps)

def batch_predict_from_torch_text(model, task, preprocess=None, postprocess=None, batch_size=16): 
    """ Batch predict function for a pytorch model """
    def batch_predict(inp):
        bsz_total = len(inp)
        # import pdb
        # pdb.set_trace()
        model.eval()
        if preprocess is not None: 
            inp = preprocess(inp) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('model.device', model.device)
        # model = model.to(device)
        # print('model.device', model.device)
        # inp = inp.to(device)
        inp = {k: torch.tensor(v) for k, v in inp.items()}

        with torch.no_grad():
            pred = []
            for b_i in tqdm(range(math.ceil(bsz_total / batch_size))):
                inp_i = {k: v[b_i*batch_size:(b_i + 1)*batch_size].to(device) for k, v in inp.items()}
                # print('cc', "inp_i['input_ids'].shape", inp_i['input_ids'].shape)
                # import pdb
                # pdb.set_trace()
                try:
                    pred_i = model(**inp_i)
                except:
                    import pdb
                    pdb.set_trace()
                if postprocess is not None:
                    pred_i = postprocess(pred_i)
                pred.append(pred_i)
            pred = torch.cat(pred)
        if task == 'reg': 
            output = pred
        elif task == 'clf': 
            output = F.softmax(pred, dim=1)
        else: 
            raise ValueError(f"Task {task} not implemented")
        return output.detach().cpu().numpy()
#         assert False
    return batch_predict

def explain_torch_reg_with_lime_text(X, model, tokenizer, label, postprocess=None,
                                normalize=False, 
                                LimeTextExplainerKwargs={}, 
                                explainer=None,
                                explain_instance_kwargs={}, 
                                get_image_and_mask_kwargs={},
                                task='clf',
                                batch_size=16): 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if explainer is None:
        explainer = lime_text.LimeTextExplainer(**LimeTextExplainerKwargs)
    # collapse = (X.ndim == 4) and (X.size(1) == 1) # check if single or RGB channel
    # X_min, X_max = X.min(), X.max()
    # if normalize: 
    #     X = (X - X_min)/(X_max-X_min) # shift to 0-1 range
    # X_np = torch_img_to_np(X.cpu()) # rearrange dimensions for numpy
    # X_np = X.cpu().numpy()
    # if collapse: 
    #     X_np = X_np[:,:,:,0] # lime needs no singleton last dimension
    # print('a')
    def p(X):
        # X = np_to_torch_img(X).to(device)
        # X = torch.from_numpy(X).to(device)
        # if collapse: 
        #     X = X[:,0:1,:,:] # even though lime needs no singleton last dimension in its input, 
        #     # for an odd reason they put back 3 of them to match RGB format before passing 
        #     # to batch_predict. So we need to remove the extraneous ones. 

        # if normalize: 
        #     X = X*(X_max - X_min) + X_min # undo shift
        # print('tok 0')
        inputs = tokenizer(X, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=512)
        # print('tok 1')
        return inputs
        
    f = batch_predict_from_torch_text(model, task, preprocess=p, 
                                      postprocess=postprocess, batch_size=batch_size)
    print('b')
    
    masks,lime_exps = [],[]
    for i, X0 in enumerate(X): 
        # print('c')
        # explainer = lime_text.LimeTextExplainer(**LimeTextExplainerKwargs)
        # print('d')
        # import pdb
        # pdb.set_trace()
        explanation = explainer.explain_instance(X0, f, **explain_instance_kwargs)
        # print('e')
        # print('label', label)
        # print('explanation.top_labels[0]', explanation.top_labels[0])
        # import pdb
        # pdb.set_trace()
        explanation_dict = {w: s for w, s in explanation.as_list()}
        # print('f')
        words = X0.split()
        # for word, weight in explanation.as_list():
        #     idx = X0.split().index(word)
        #     mask[idx] = 1
        mask = [explanation_dict[word] if word in explanation_dict else 0 for word in words]
        mask = F.pad(torch.tensor(mask), (0, 512 - len(mask)))
        # print('g')
        # import pdb
        # pdb.set_trace()
        # print('explanation', explanation)
        # img,mask = explanation.get_image_and_mask(explanation.top_labels[0] 
        #                                           if label is None else label[i].cpu().numpy().item(), 
        #                                           **get_image_and_mask_kwargs)

        masks.append(mask)
        lime_exps.append(explanation)
        # print('h')
    masks = torch.stack(masks)
    # print('i')

    return AttributionOutput(masks, lime_exps)
