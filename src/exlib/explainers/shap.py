import torch
import torch.nn.functional as F
import shap
from .common import AttributionOutput, torch_img_to_np, np_to_torch_img

def explain_torch_with_shap(X, model, mask_value, explainer_kwargs, 
                            shap_kwargs, postprocess=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_np = torch_img_to_np(X.cpu())
    masker = shap.maskers.Image(mask_value, X_np[0].shape)

    def f(X): 
        model.to(device)
        with torch.no_grad(): 
            pred = model(np_to_torch_img(X).to(device))
            if postprocess:
                pred = postprocess(pred)
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(f, masker, **explainer_kwargs)

    # here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X_np, **shap_kwargs)

    if shap_values.values.shape[-1] == 1: 
        sv = np_to_torch_img(shap_values.values[:,:,:,:,0])
        return AttributionOutput(sv, shap_values)
    else: 
        raise ValueError("Not implemented for explaining more than one output")
    
def explain_torch_with_shap_text(X, model, tokenizer, mask_value, explainer_kwargs, 
                            shap_kwargs, postprocess=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # X_np = torch_img_to_np(X.cpu())
    # X_np = X.cpu().numpy()
    # kwargs_np = {k: v.cpu().numpy() for k, v in kwargs.items()}
    # kwargs_np['input_ids'] = X_np
    # masker = shap.maskers.Text(mask_token=mask_value)
    inputs = tokenizer(X, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=512)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    pred = model(**inputs)
    if postprocess:
        pred = postprocess(pred)
    _, predicted = torch.max(pred, -1)

    def f(X): 
        # passage = [x.split('\t')[0] for x in X]
        # query_and_answer = [x.split('\t')[1] for x in X]  #texts = [x.split('\t') for x in X]
        model.to(device)
        with torch.no_grad(): 
            inputs = tokenizer(X.tolist(), 
                                padding='max_length', 
                                truncation=True, 
                                max_length=512)
            inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
            pred = model(**inputs)
            # kwargs_pt = {k: torch.from_numpy(v) for k, v in X.items()}
            # pred = model(np_to_torch_img(X).to(device))

            if postprocess:
                pred = postprocess(pred)
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(f, tokenizer)

    # here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X)
    # shap_values = explainer(X_np, **shap_kwargs)

    if shap_values.values.shape[-1] == 1: 
        sv = torch.from_numpy(shap_values.values[:,:,:,:,0])
        # sv = np_to_torch_img(shap_values.values[:,:,:,:,0])
        return AttributionOutput(sv, shap_values)
    else: 
        predicted_labels = predicted.cpu().numpy()
        svs = [torch.tensor(sv[:,predicted_labels[sv_i]]) for sv_i, sv in enumerate(shap_values.values)]
        
        def pad_tensor_to_length(tensor, target_length=512, pad_value=0):
            """Pad tensor with pad_value up to target_length."""
            pad_length = target_length - tensor.size(0)
            return F.pad(tensor, (0, pad_length), 'constant', pad_value)
        # Pad each tensor in the list
        padded_svs = [pad_tensor_to_length(tensor) for tensor in svs]

        # Combine the padded tensors into a single tensor
        combined_svs = torch.stack(padded_svs, dim=0)

        return AttributionOutput(combined_svs, shap_values)
        # raise ValueError("Not implemented for explaining more than one output")
    