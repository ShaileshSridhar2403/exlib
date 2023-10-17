# TODO: optimize

from __future__ import division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
import collections.abc
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PretrainedConfig, PreTrainedModel
import copy
import json
from collections import namedtuple


AttributionOutputSOP = namedtuple("AttributionOutputSOP", 
                                  ["logits",
                                   "logits_all",
                                   "pooler_outputs_all",
                                   "masks",
                                   "mask_weights",
                                   "attributions", 
                                   "attributions_max",
                                   "attributions_all",
                                   "flat_masks"])


def get_chained_attr(obj, attr_chain):
    attrs = attr_chain.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def compress_single_masks(masks, masks_weights, min_size):
    # num_masks, seq_len = masks.shape
    masks_bool = (masks > 0).int()
    sorted_weights, sorted_indices = torch.sort(masks_weights, descending=True)
    sorted_indices = sorted_indices[sorted_weights > 0]

    masks_bool = masks_bool[sorted_indices]  # sorted masks
    
    masks = torch.zeros(*masks_bool.shape[1:]).to(masks.device)
    count = 1
    for mask in masks_bool:
        new_mask = mask.bool() ^ (mask.bool() & masks.bool())
        if torch.sum(new_mask) >= min_size:
            masks[new_mask] = count
            count += 1

    masks = masks - 1
    masks = masks.int()
    masks[masks == -1] = torch.max(masks) + 1

    return masks

def compress_masks(masks, masks_weights, min_size=0):
    new_masks = []
    for i in range(len(masks)):
        compressed_mask = compress_single_masks(masks[i], masks_weights[i], 
                                                min_size)
        new_masks.append(compressed_mask)
    return torch.stack(new_masks)


def compress_masks_image(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 4 # bsz, num_masks, img_dim_1, img_dim_2 = masks.shape
    return compress_masks(masks, masks_weights, min_size)
    

def compress_masks_text(masks, masks_weights, min_size=0):
    assert len(masks.shape) == 3 # bsz, num_masks, seq_len = masks.shape
    return compress_masks(masks, masks_weights, min_size)
           

def _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, 
    num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, 
):
    epoch = current_step // (num_steps_per_epoch * num_heads)
    steps_within_epoch = current_step % num_steps_per_epoch
    step_for_curr_head = epoch * num_steps_per_epoch + steps_within_epoch
    if step_for_curr_head < num_warmup_steps:
        return float(step_for_curr_head) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((step_for_curr_head + shift) / timescale)
    return decay

def get_inverse_sqrt_with_separate_heads_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_steps_per_epoch: int,
    timescale: int = None, 
    num_heads: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if timescale is None:
        timescale = num_warmup_steps

    lr_lambda = partial(
        _get_inverse_sqrt_with_separate_heads_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_steps_per_epoch=num_steps_per_epoch,
        timescale=timescale,
        num_heads=num_heads,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


"""Sparsemax activation function.

Pytorch implementation of Sparsemax function from:
-- "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
-- André F. T. Martins, Ramón Fernandez Astudillo (http://arxiv.org/abs/1602.02068)
"""


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, inputs):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        device = inputs.device
        inputs = inputs.transpose(0, self.dim)
        original_size = inputs.size()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs.transpose(0, 1)
        dim = 1

        number_of_logits = inputs.size(dim)

        # Translate input by max for numerical stability
        inputs = inputs - torch.max(inputs, dim=dim, keepdim=True)[0].expand_as(inputs)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=inputs, dim=dim, descending=True)[0]
        range_tensor = torch.arange(start=1, end=number_of_logits + 1, step=1, 
                                    device=device, dtype=inputs.dtype).view(1, -1)
        range_tensor = range_tensor.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(inputs.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(inputs)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(inputs), inputs - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input
    

class GroupGenerateLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
       
        # self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, 
        #                                             batch_first=True)
        self.multihead_attns = nn.ModuleList([nn.MultiheadAttention(hidden_dim, 
                                                                   1, 
                                                                   batch_first=True) \
                                                for _ in range(num_heads)])
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, query, key_value, epoch=0):
        """
            Use multiheaded attention to get mask
            Num_interpretable_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """
        epsilon = 1e-30

        head_i = epoch % self.num_heads
        if self.training:
            _, attn_weights = self.multihead_attns[head_i](query, key_value, key_value, 
                                                          average_attn_weights=False)
        else:
            attn_weights = []
            if epoch < self.num_heads:
                num_heads_use = head_i + 1
            else:
                num_heads_use = self.num_heads
            for head_j in range(num_heads_use):
                _, attn_weights_j = self.multihead_attns[head_j](query, key_value, key_value)
                attn_weights.append(attn_weights_j)
            attn_weights = torch.stack(attn_weights, dim=1)
        
        attn_weights = attn_weights + epsilon
        mask = self.sparsemax(torch.log(attn_weights))
            
        return mask


class GroupSelectLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        # self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, 1, 
                                                    batch_first=True)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, query, key, value):
        """
            Use multiheaded attention to get mask
            Num_heads = num_heads * seq_len
            Input: x (bsz, seq_len, hidden_dim)
                   if actual_x is not None, then use actual_x instead of x to compute attn_output
            Output: attn_outputs (bsz, num_heads * seq_len, seq_len, hidden_dim)
                    mask (bsz, num_heads, seq_len, seq_len)
        """
        # x shape: (batch_size, sequence_length, hidden_dim)
        # x shape: (..., hidden_dim)
        epsilon = 1e-30
        bsz, seq_len, hidden_dim = query.shape

        # Obtain attention weights
        _, attn_weights = self.multihead_attn(query, key, key)
        attn_weights = attn_weights + epsilon  # (batch_size, num_heads, sequence_length, hidden_dim)
        # attn_output: (batch_size, sequence_length, hidden_dim)
        # attn_weights: (batch_size, num_heads, sequence_length, hidden_dim)
        # attn_weights = attn_weights.mean(dim=-2)  # average seq_len number of heads
        # if we do sparsemax directly, they are all within 0 and 1 and thus don't move. 
        # need to first transform to log space.
        mask = self.sparsemax(torch.log(attn_weights))
        mask = mask.transpose(-1, -2)

        # Apply attention weights on what to be attended
        new_shape = list(mask.shape) + [1] * (len(value.shape) - 3)
        attn_outputs = (value * mask.view(*new_shape)).sum(1)

        # attn_outputs of shape (bsz, num_masks, num_classes)
        return attn_outputs, mask


class SOPConfig(PretrainedConfig):
    def __init__(self,
                 json_file=None,
                 hidden_size: int = 512,
                 num_labels: int = 2,
                 projected_input_scale: int = 1,
                 num_heads: int = 1,
                 num_masks_sample: int = 20,
                 num_masks_max: int = 200,
                 image_size=(224, 224),
                 num_channels: int = 3,
                 attn_patch_size: int = 16,
                 finetune_layers=[],
                 **kwargs):

        # all the config from the json file will be in self.__dict__
        super().__init__(**kwargs)

        if json_file is not None:
            self.update_from_json(json_file)
        
        # overwrite the config from json file if specified
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.projected_input_scale = projected_input_scale
        self.num_heads = num_heads
        self.num_masks_sample = num_masks_sample
        self.num_masks_max = num_masks_max
        self.image_size = image_size
        self.num_channels = num_channels
        self.attn_patch_size = attn_patch_size
        self.finetune_layers = finetune_layers
        
        
    
    def update_from_json(self, json_file):
        with open(json_file, 'r') as f:
            json_dict = json.load(f)
        self.__dict__.update(json_dict)


class SOP(PreTrainedModel):
    config_class = SOPConfig

    def __init__(self, 
                 config,
                 backbone_model,
                 class_weights
                 ):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size  # match black_box_model hidden_size
        self.num_classes = config.num_labels if hasattr(config, 'num_labels') is not None else 1  # 1 is for regression
        self.projected_input_scale = config.projected_input_scale if hasattr(config, 'projected_input_scale') else 1
        self.num_heads = config.num_heads
        self.num_masks_sample = config.num_masks_sample
        self.num_masks_max = config.num_masks_max
        self.finetune_layers = config.finetune_layers

        # blackbox model and finetune layers
        self.blackbox_model = backbone_model
        self.class_weights = copy.deepcopy(class_weights.detach())
        
        
        self.input_attn = GroupGenerateLayer(hidden_dim=self.hidden_size,
                                             num_heads=self.num_heads)
        # output
        self.output_attn = GroupSelectLayer(hidden_dim=self.hidden_size)

    def init_grads(self):
        # Initialize the weights of the model
        # blackbox_model = self.blackbox_model.clone()
        # self.init_weights()
        # self.blackbox_model = blackbox_model

        for name, param in self.blackbox_model.named_parameters():
            param.requires_grad = False

        for finetune_layer in self.finetune_layers:
            # todo: check if this works with index
            for name, param in get_chained_attr(self.blackbox_model, finetune_layer).named_parameters():
                param.requires_grad = True

    def forward(self):
        raise NotImplementedError


# class SOPModel(PreTrainedModel):
#     config_class = SOPConfig

#     def __init__(self, config, backbone_model, class_weights):
#         super().__init__(config)
#         self.sop_model = SOP(
#             config=config,
#             backbone_model=backbone_model,
#             class_weights=class_weights
#         )

#     def forward(self, **inputs):
#         return self.model.forward(**inputs)


class SOPImage(SOP):
    def __init__(self, 
                 config,
                 blackbox_model,
                 class_weights,
                 projection_layer=None,
                 ):
        super().__init__(config,
                            blackbox_model,
                            class_weights
                            )
        
        self.image_size = config.image_size if isinstance(config.image_size, 
                                                    collections.abc.Iterable) \
                                            else (config.image_size, config.image_size)
        self.num_channels = config.num_channels
        # attention args
        self.attn_patch_size = config.attn_patch_size

        if projection_layer is not None:
            self.projection = copy.deepcopy(projection_layer)
        else:
            self.init_projection()

        # Initialize the weights of the model
        self.init_weights()
        self.init_grads()

    def init_projection(self):
        self.projection = nn.Conv2d(self.config.num_channels, 
                                    self.hidden_size, 
                                    kernel_size=self.attn_patch_size, 
                                    stride=self.attn_patch_size)  # make each patch a vec
        self.projection_up = nn.ConvTranspose2d(1, 
                                                    1, 
                                                    kernel_size=self.attn_patch_size, 
                                                    stride=self.attn_patch_size)  # make each patch a vec
        self.projection_up.weight = nn.Parameter(torch.ones_like(self.projection_up.weight))
        self.projection_up.bias = nn.Parameter(torch.zeros_like(self.projection_up.bias))
        self.projection_up.weight.requires_grad = False
        self.projection_up.bias.requires_grad = False

    def forward(self, 
                inputs, 
                segs=None, 
                input_mask_weights=None,
                epoch=-1, 
                mask_batch_size=16,
                label=None,
                return_tuple=False):
        if epoch == -1:
            epoch = self.num_heads
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        
        # Mask (Group) generation
        if input_mask_weights is None:
            grouped_inputs, input_mask_weights = self.group_generate(inputs, epoch, mask_batch_size, segs)
        else:
            grouped_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2) # directly apply mask
        
        # Backbone model
        logits, pooler_outputs = self.run_backbone(grouped_inputs, mask_batch_size)
        # return logits

        # Mask (Group) selection & aggregation
        # return self.group_select(logits, pooler_outputs, img_dim1, img_dim2)
        weighted_logits, output_mask_weights, logits, pooler_outputs = self.group_select(logits, pooler_outputs, img_dim1, img_dim2)

        if return_tuple:
            return self.get_results_tuple(weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label)
        else:
            return weighted_logits

    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        raise NotImplementedError

    def run_backbone(self, masked_inputs, mask_batch_size):
        bsz, num_masks, num_channel, img_dim1, img_dim2 = masked_inputs.shape
        masked_inputs = masked_inputs.view(-1, num_channel, img_dim1, img_dim2)
        logits = []
        pooler_outputs = []
        for i in range(0, masked_inputs.shape[0], mask_batch_size):
            output_i = self.blackbox_model(
                masked_inputs[i:i+mask_batch_size]
            )
            pooler_i = output_i.pooler_output
            logits_i = output_i.logits
            logits.append(logits_i)
            pooler_outputs.append(pooler_i)

        logits = torch.cat(logits).view(bsz, num_masks, self.num_classes, -1)
        pooler_outputs = torch.cat(pooler_outputs).view(bsz, num_masks, self.hidden_size, -1)
        return logits, pooler_outputs
    
    def group_generate(self, inputs, epoch, mask_batch_size, segs=None):
        bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        if segs is None:   # should be renamed "segments"
            projected_inputs = self.projection(inputs)
            projected_inputs = projected_inputs.flatten(2).transpose(1, 2)  # bsz, img_dim1 * img_dim2, num_channel
            projected_inputs = projected_inputs * self.projected_input_scale

            if self.num_masks_max != -1:
                input_dropout_idxs = torch.randperm(projected_inputs.shape[1])[:self.num_masks_max]
                projected_query = projected_inputs[:, input_dropout_idxs]
            else:
                projected_query = projected_inputs
            input_mask_weights_cand = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            
            num_patches = ((self.image_size[0] - self.attn_patch_size) // self.attn_patch_size + 1, 
                        (self.image_size[1] - self.attn_patch_size) // self.attn_patch_size + 1)
            input_mask_weights_cand = input_mask_weights_cand.reshape(-1, 1, num_patches[0], num_patches[1])
            input_mask_weights_cand = self.projection_up(input_mask_weights_cand, 
                                                         output_size=torch.Size([input_mask_weights_cand.shape[0], 1, 
                                                                                 img_dim1, img_dim2]))
            input_mask_weights_cand = input_mask_weights_cand.view(bsz, -1, img_dim1, img_dim2)
            input_mask_weights_cand = torch.clip(input_mask_weights_cand, max=1.0)
        else:
            # With/without masks are a bit different. Should we make them the same? Need to experiment.
            bsz, num_segs, img_dim1, img_dim2 = segs.shape
            seged_output_0 = inputs.unsqueeze(1) * segs.unsqueeze(2) # (bsz, num_masks, num_channel, img_dim1, img_dim2)
            interm_outputs, _ = self.run_backbone(seged_output_0, mask_batch_size)
            
            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            interm_outputs = interm_outputs * self.projected_input_scale
            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_segs)
            
            new_masks =  segs.unsqueeze(1) * segment_mask_weights.unsqueeze(-1).unsqueeze(-1)
            # (bsz, num_new_masks, num_masks, img_dim1, img_dim2)
            input_mask_weights_cand = new_masks.sum(2)  # if one mask has it, then have it
            # todo: Can we simplify the above to be dot product?
            
        scale_factor = 1.0 / input_mask_weights_cand.reshape(bsz, -1, 
                                                        img_dim1 * img_dim2).max(dim=-1).values
        input_mask_weights_cand = input_mask_weights_cand * scale_factor.view(bsz, -1,1,1)
        
        
        # we are using iterative training
        # we will train some masks every epoch
        # the masks to train are selected by mod of epoch number
        # Dropout for training
        if self.training:
            dropout_idxs = torch.randperm(input_mask_weights_cand.shape[1])[:self.num_masks_sample]
            dropout_mask = torch.zeros(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
            dropout_mask[:,dropout_idxs] = 1
        else:
            dropout_mask = torch.ones(bsz, input_mask_weights_cand.shape[1]).to(inputs.device)
        
        input_mask_weights = input_mask_weights_cand[dropout_mask.bool()].clone()
        input_mask_weights = input_mask_weights.view(bsz, -1, img_dim1, img_dim2)

        masked_inputs = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2)
        return masked_inputs, input_mask_weights
    
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        raise NotImplementedError

class SOPImageCls(SOPImage):
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        bsz, num_masks = logits.shape[:2]

        logits = logits.view(bsz, num_masks, self.num_classes)
        pooler_outputs = pooler_outputs.view(bsz, num_masks, self.hidden_size)

        query = self.class_weights.unsqueeze(0).expand(bsz, 
                                                    self.num_classes, 
                                                    self.hidden_size) #.to(logits.device)
       
        key = pooler_outputs
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)
        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        # todo: debug for segmentation
        masks_aggr = None
        masks_aggr_pred_cls = None
        masks_max_pred_cls = None
        flat_masks = None

        if label is not None:
            predicted = label  # allow labels to be different
        else:
            _, predicted = torch.max(weighted_logits.data, -1)
        
        masks_mult = input_mask_weights.unsqueeze(2) * \
        output_mask_weights.unsqueeze(-1).unsqueeze(-1) # bsz, n_masks, n_cls, img_dim, img_dim
        
        masks_aggr = masks_mult.sum(1) # bsz, n_cls, img_dim, img_dim OR bsz, n_cls, seq_len
        masks_aggr_pred_cls = masks_aggr[range(bsz), predicted].unsqueeze(1)
        max_mask_indices = output_mask_weights.max(2).values.max(1).indices
        masks_max_pred_cls = masks_mult[range(bsz),max_mask_indices,predicted].unsqueeze(1)
        flat_masks = compress_masks_image(input_mask_weights, output_mask_weights)
        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks)
    

class SOPImageSeg(SOPImage):
    def group_select(self, logits, pooler_outputs, img_dim1, img_dim2):
        bsz, num_masks = logits.shape[:2]
        logits = logits.view(bsz, num_masks, self.num_classes, 
                                            img_dim1, img_dim2)
        pooler_outputs = pooler_outputs.view(bsz, num_masks, 
                                                        self.hidden_size, 
                                                        img_dim1, 
                                                        img_dim2)
        # return pooler_outputs
        query = self.class_weights.unsqueeze(0) \
            .view(1, self.num_classes, self.hidden_size, -1).mean(-1) \
            .expand(bsz, self.num_classes, self.hidden_size).to(logits.device)
        pooler_outputs.requires_grad = True
        key = pooler_outputs.view(bsz, num_masks, self.hidden_size, -1).mean(-1)
        weighted_logits, output_mask_weights = self.output_attn(query, key, logits)
        return weighted_logits, output_mask_weights, logits, pooler_outputs
    
    def get_results_tuple(self, weighted_logits, logits, pooler_outputs, input_mask_weights, output_mask_weights, bsz, label):
        # todo: debug for segmentation
        masks_aggr = None
        masks_aggr_pred_cls = None
        masks_max_pred_cls = None
        flat_masks = None

        # import pdb
        # pdb.set_trace()
        # _, predicted = torch.max(weighted_output.data, -1)
        masks_mult = input_mask_weights.unsqueeze(2) * output_mask_weights.unsqueeze(-1).unsqueeze(-1) # bsz, n_masks, n_cls, img_dim, img_dim
        masks_aggr = masks_mult.sum(1) # bsz, n_cls, img_dim, img_dim OR bsz, n_cls, seq_len
        # masks_aggr_pred_cls = masks_aggr
        # masks_aggr_pred_cls = masks_aggr[range(bsz), predicted].unsqueeze(1)
        # max_mask_indices = output_mask_weights.argmax(1)
        # masks_max_pred_cls = masks_mult[max_mask_indices[:,0]]
        # masks_max_pred_cls = max_mask_indices
        # TODO: this has some problems ^
        # import pdb
        # pdb.set_trace()
        
        flat_masks = compress_masks_image(input_mask_weights, output_mask_weights)

        return AttributionOutputSOP(weighted_logits,
                                    logits,
                                    pooler_outputs,
                                    input_mask_weights,
                                    output_mask_weights,
                                    masks_aggr_pred_cls,
                                    masks_max_pred_cls,
                                    masks_aggr,
                                    flat_masks)
