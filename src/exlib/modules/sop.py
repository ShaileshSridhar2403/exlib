from __future__ import division
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import collections.abc
from functools import partial
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import copy
# from pathlib import Path
# import sys
# print(Path(__file__).parents[0])
# print(Path(__file__).parents[1])
# path_root = Path(__file__).parents[1]
# print(path_root)
# sys.path.append(str(path_root))
from collections import namedtuple

AttributionOutputSOP = namedtuple("AttributionOutputSOP", 
                                  ["attributions", 
                                   "logits",
                                   "logits_all",
                                   "seg_weights",
                                   "mask_weights",
                                   "attributions_max"])


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



def log_softmax_score_vectorized_batched(attn, logits):
    """
    Compute scores to maximize each class's probability using 
    a combination of logits from different masks.
    @param attn: (bsz, num_masks, num_classes)
    @param logits: (bsz, num_masks, num_classes)
    return: 
    """
    log_scores = []

    log_scores_curr = torch.sum(attn * logits, dim=1)
    log_scores_support = torch.logsumexp(attn.transpose(1, 2).bmm(logits), -1)
    log_scores = log_scores_curr - log_scores_support

    return log_scores  # (bsz, num_classes)


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
    

class SparsemaxMaskExplanationLayer(nn.Module):
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
            attn_weights = torch.cat(attn_weights, dim=1)
        
        # import pdb
        # pdb.set_trace()
        attn_weights = attn_weights + epsilon
        mask = self.sparsemax(torch.log(attn_weights))
        # import pdb
        # pdb.set_trace()
            
        return mask


class AggregatePerClassAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, aggr_type='joint'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, 
                                                    batch_first=True)
        self.sparsemax = Sparsemax(dim=-1)
        self.aggr_type = aggr_type
        if aggr_type not in ['joint', 'independent']:
            raise ValueError('Aggr_type needs  to be one of joint or independent')

    def forward(self, query, key_value, to_attend):
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
        _, attn_weights = self.multihead_attn(query, key_value, key_value)
        attn_weights = attn_weights + epsilon
        # attn_output: (batch_size, sequence_length, hidden_dim)
        # attn_weights: (batch_size, num_heads, sequence_length, hidden_dim)
        # attn_weights = attn_weights.mean(dim=-2)  # average seq_len number of heads
        # if we do sparsemax directly, they are all within 0 and 1 and thus don't move. 
        # need to first transform to log space.
        mask = self.sparsemax(torch.log(attn_weights))
        mask = mask.transpose(-1, -2)

        # Apply attention weights on what to be attended
        if self.aggr_type == 'joint':
            attn_outputs = log_softmax_score_vectorized_batched(mask, to_attend)
        else:  # independent
            if len(to_attend.shape) == 5:
                attn_outputs = to_attend * mask.unsqueeze(-1).unsqueeze(-1)
            else:
                attn_outputs = to_attend * mask  # .view(1, -1, 1)
            attn_outputs = attn_outputs.sum(1)

        # attn_outputs of shape (bsz, num_masks, num_classes)
        return attn_outputs, mask


class SOP(PreTrainedModel):
    def __init__(self, 
                 config,
                 blackbox_model,
                 model_type='image',
                 projection_layer=None,
                 aggr_type='joint',
                 pooler=True,
                 proj_hid_size=None,
                 mean_center_scale=0,
                 mean_center_scale2=0,
                 pred_type='sequence',
                 mean_center_offset=0,
                 mean_center_offset2=None,
                 return_tuple=False
                 ):
        if config is not None:
            super().__init__(config)
        else:
            super().__init__()
        self.config = config
        self.model_type = model_type
        self.return_tuple = return_tuple

        # need from original model's config:
        self.hidden_size = config.hidden_size  # match black_box_model hidden_size
        if model_type == 'image':
            self.image_size = config.image_size if isinstance(config.image_size, 
                                                        collections.abc.Iterable) \
                                                else (config.image_size, config.image_size)
            self.num_channels = config.num_channels
        else:  # text
            self.image_size = None
            self.num_channels = None
        # print('config.num_labels', config.num_labels)
        # import pdb
        # pdb.set_trace()
        self.num_classes = config.num_labels if config.num_labels is not None else 1  # 1 is for regression
        
        self.pooler = pooler

        # attention args
        if model_type == 'image':
            self.attn_patch_size = config.attn_patch_size
            if hasattr(config, 'attn_stride_size') and \
                config.attn_stride_size is not None:
                self.attn_stride_size = config.attn_stride_size
            else:
                self.attn_stride_size = config.attn_patch_size
        else:
            self.attn_patch_size = None
            self.attn_stride_size = None
        if proj_hid_size is not None:
            self.proj_hid_size = proj_hid_size
        else:
            self.proj_hid_size = config.hidden_size
        self.mean_center_scale = mean_center_scale
        self.mean_center_scale2 = mean_center_scale2
        self.mean_center_offset = mean_center_offset
        self.mean_center_offset2 = mean_center_offset2
        self.num_heads = config.num_heads
        # self.attn_hidden_multiple = config.attn_hidden_multiple
        # self.output_attn_hidden_dim = output_attn_hidden_dim
        self.num_masks_sample = config.num_masks_sample
        self.num_masks_max = config.num_masks_max
        self.pred_type = pred_type  # sequence, token
        # self.output_aggr_type = output_aggr_type

        # blackbox model and finetune layers
        self.blackbox_model = blackbox_model
        self.finetune_layers = config.finetune_layers # e.g. ['classifier', 'fc']
        if hasattr(config, 'finetune_layers_idxs'):
            self.finetune_layers_idxs = config.finetune_layers_idxs
        else:
            self.finetune_layers_idxs = None

        # attention - new parts
        # input
        self.projection_layer = projection_layer
        if projection_layer is not None:
            self.projection = copy.deepcopy(projection_layer)
        elif model_type == 'image':
            self.projection = nn.Conv2d(config.num_channels, 
                                        self.proj_hid_size, 
                                        kernel_size=self.attn_patch_size, 
                                        stride=self.attn_stride_size)  # make each patch a vec
            self.projection_up = nn.ConvTranspose2d(1, 
                                                      1, 
                                                      kernel_size=self.attn_patch_size, 
                                                      stride=self.attn_stride_size)  # make each patch a vec
            self.projection_up.weight = nn.Parameter(torch.ones_like(self.projection_up.weight))
            self.projection_up.bias = torch.nn.Parameter(torch.zeros_like(self.projection_up.bias))
            self.projection_up.weight.requires_grad = False
            self.projection_up.bias.requires_grad = False
        else:  # text
            self.projection = nn.Linear(1, self.proj_hid_size)
        self.input_attn = SparsemaxMaskExplanationLayer(hidden_dim=self.proj_hid_size,
                                                        num_heads=self.num_heads)
        # output
        self.output_attn = AggregatePerClassAttentionLayer(hidden_dim=self.hidden_size,
                                            num_heads=1,
                                            aggr_type=aggr_type)

        # Initialize the weights of the model
        self.init_weights()
        self.blackbox_model = blackbox_model
        if self.finetune_layers_idxs is None:
            self.class_weights = copy.deepcopy(getattr(self.blackbox_model, self.finetune_layers[0]).weight)
            # Freeze the frozen module
            for name, param in self.blackbox_model.named_parameters():
                if sum([ft_layer in name for ft_layer in self.finetune_layers]) == 0: # the name doesn't match any finetune layers
                    param.requires_grad = False
        else:
            self.class_weights = copy.deepcopy(getattr(self.blackbox_model, self.finetune_layers[0])[self.finetune_layers_idxs[0]].weight)
            # Freeze the frozen module
            for name, param in self.blackbox_model.named_parameters():
                if sum([f'{self.finetune_layers[i]}.{self.finetune_layers_idxs[i]}' in name for i in range(len(self.finetune_layers))]) == 0: # the name doesn't match any finetune layers
                    param.requires_grad = False

    def forward(self, inputs, masks=None, epoch=-1, mask_batch_size=16):
        if epoch == -1:
            epoch = self.num_heads
        if self.model_type == 'image':
            bsz, num_channel, img_dim1, img_dim2 = inputs.shape
        else:
            bsz, seq_len = inputs.shape
        
        # Input masks
        # print('inputs.shape', inputs.shape)
        # import pdb
        # pdb.set_trace()
        if masks is None:
            centered_inputs = inputs + self.mean_center_offset
            if self.mean_center_scale > 0:
                centered_inputs = centered_inputs * self.mean_center_scale  # + bc msc is neg
            else:
                centered_inputs = centered_inputs
            projected_inputs = self.projection(centered_inputs)
            # print('projected_inputs', projected_inputs)
            # import pdb
            # pdb.set_trace()
            
            if self.model_type == 'image':
                projected_inputs = projected_inputs.flatten(2).transpose(1, 2)  # bsz, img_dim1 * img_dim2, num_channel
            
            if self.mean_center_scale2 > 0:
                # projected_inputs = (projected_inputs - projected_inputs.mean(-2)) * 100 # trying to see if this can help
                if self.mean_center_offset2 is None:
                    projected_inputs = (projected_inputs - projected_inputs.mean(-2).unsqueeze(-2)) * 1 / \
                        projected_inputs.mean(-2).unsqueeze(-2) * self.mean_center_scale2
                else:
                    projected_inputs = (projected_inputs + self.mean_center_offset2) * self.mean_center_scale2
            # elif self.mean_center_scale < 0:
            #     projected_inputs = (projected_inputs - projected_inputs.mean(-2).unsqueeze(-2)) * 1 / projected_inputs.mean(-2).unsqueeze(-2) / (- self.mean_center_scale)

            if self.num_masks_max != -1:
                input_dropout_idxs = torch.randperm(projected_inputs.shape[1])[:self.num_masks_max]
                projected_query = projected_inputs[:, input_dropout_idxs]
            else:
                projected_query = projected_inputs
            input_mask_weights = self.input_attn(projected_query, projected_inputs, epoch=epoch)
            # print('input_mask_weights a', input_mask_weights.shape)
            # import pdb
            # pdb.set_trace()
            num_patches = ((self.image_size[0] - self.attn_patch_size) // self.attn_stride_size + 1, 
                        (self.image_size[1] - self.attn_patch_size) // self.attn_stride_size + 1)
            input_mask_weights = input_mask_weights.reshape(-1, 1, num_patches[0], num_patches[1])
            input_mask_weights = self.projection_up(input_mask_weights, output_size=torch.Size([input_mask_weights.shape[0], 1, img_dim1, img_dim2]))
            input_mask_weights = input_mask_weights.view(bsz, -1, img_dim1, img_dim2)
            input_mask_weights = torch.clip(input_mask_weights, max=1.0)

            # num_patches = (self.image_size[0] // self.attn_patch_size, 
            #                 self.image_size[1] // self.attn_patch_size)
            # input_mask_weights = input_mask_weights.view(bsz, -1, 
            #                                                 num_patches[0], 
            #                                                 num_patches[1]).unsqueeze(-1).unsqueeze(-3)\
            #     .expand(bsz, -1, num_patches[0], self.attn_patch_size, 
            #             num_patches[1], self.attn_patch_size)\
            #     .contiguous().view(bsz, -1, img_dim1, img_dim2)
            # print('input_mask_weights b', input_mask_weights.shape)
            # import pdb
            # pdb.set_trace()
            # input_mask_weights: (bsz, seq_len, img_dim1, img_dim2)
            # scale the attention weights to be max 1 for each mask.
            
        else:
            bsz, num_masks, img_dim1, img_dim2 = masks.shape
            masked_output_0 = inputs.unsqueeze(1) * masks.unsqueeze(2)
            # (bsz, num_masks, num_channel, img_dim1, img_dim2)
            masked_output_0 = masked_output_0.view(-1, num_channel, img_dim1, img_dim2)
            interm_outputs = []
            for i in range(0, masked_output_0.shape[0], mask_batch_size):
                # output_i = self.blackbox_model(
                #     masked_output_0[i:i+self.mask_batch_size],
                #     output_hidden_states=True
                # ).hidden_states[-1]

                if self.pooler:
                    output_i = self.blackbox_model(
                        masked_output_0[i:i+mask_batch_size]
                    )
                    pooler_i = output_i.pooler_output
                else:
                    output_i = self.blackbox_model(
                        masked_output_0[i:i+mask_batch_size],
                        output_hidden_states=True
                    )
                    pooler_i = output_i.hidden_states[-1][:,0]
                # logits_i = output_i.logits
                
                interm_outputs.append(pooler_i) # only use cls head
            
            interm_outputs = torch.cat(interm_outputs)

            interm_outputs = interm_outputs.view(bsz, -1, self.hidden_size)
            # interm_outputs = interm_outputs.unsqueeze(-2).expand(bsz, 
            #                                            -1, 
            #                                            self.num_heads1_hidden_multiple,
            #                                            self.hidden_size).reshape(bsz, 
            #                                                             -1, 
            #                                                             self.num_heads1_hidden_multiple * \
            #                                                             self.hidden_size)
            if self.mean_center_scale2 > 0:
                # projected_inputs = (projected_inputs - projected_inputs.mean(-2)) * 100 # trying to see if this can help
                if self.mean_center_offset2 is None:
                    interm_outputs = (interm_outputs - interm_outputs.mean(-2).unsqueeze(-2)) * 1 / \
                        interm_outputs.mean(-2).unsqueeze(-2) * self.mean_center_scale2
                else:
                    interm_outputs = (interm_outputs + self.mean_center_offset2) * self.mean_center_scale2

            segment_mask_weights = self.input_attn(interm_outputs, interm_outputs, epoch=epoch)
            
            segment_mask_weights = segment_mask_weights.reshape(bsz, -1, num_masks)
            
            new_masks =  masks.unsqueeze(1) * \
                segment_mask_weights.unsqueeze(-1).unsqueeze(-1)
            # (bsz, num_new_masks, num_masks, img_dim1, img_dim2)
            input_mask_weights = new_masks.sum(2)  # if one mask has it, then have it
            
        scale_factor = 1.0 / input_mask_weights.reshape(bsz, -1, 
                                                        img_dim1 * img_dim2).max(dim=-1).values
        input_mask_weights = input_mask_weights * scale_factor.view(bsz, -1,1,1)
        
            # todo: Can we simplify the above to be dot product?
        # print('input_mask_weights final', input_mask_weights)
        # print('input_mask_weights c', input_mask_weights.shape)
        # import pdb
        # pdb.set_trace()
        # we are using iterative training
        # we will train some masks every epoch
        # the masks to train are selected by mod of epoch number
        if self.training:
            dropout_idxs = torch.randperm(input_mask_weights.shape[1])[:self.num_masks_sample]
            dropout_mask = torch.zeros(bsz, input_mask_weights.shape[1])
            dropout_mask[:,dropout_idxs] = 1
        else:
            dropout_mask = torch.ones(bsz, input_mask_weights.shape[1])
        
        selected_input_mask_weights = input_mask_weights[dropout_mask.bool()].clone()
        selected_input_mask_weights = selected_input_mask_weights.view(bsz, 
                                                                       -1, 
                                                                       img_dim1, 
                                                                       img_dim2)
        
        # masked_output = inputs.unsqueeze(1) * input_mask_weights.unsqueeze(2)   # (bsz, seq_len, num_channel, img_dim1, img_dim2)
        selected_masked_inputs = inputs.unsqueeze(1) * selected_input_mask_weights.unsqueeze(2)
        selected_masked_inputs = selected_masked_inputs.view(-1, 
                                                             num_channel, 
                                                             img_dim1, 
                                                             img_dim2)

        outputs = []
        pooler_outputs = []
        for i in range(0, selected_masked_inputs.shape[0], mask_batch_size):
            if self.pooler:
                output_i = self.blackbox_model(
                    selected_masked_inputs[i:i+mask_batch_size]
                )
                pooler_i = output_i.pooler_output
            else:
                output_i = self.blackbox_model(
                    selected_masked_inputs[i:i+mask_batch_size],
                    output_hidden_states=True
                )
                pooler_i = output_i.hidden_states[-1][:,0]
            logits_i = output_i.logits
            outputs.append(logits_i)
            pooler_outputs.append(pooler_i)
        
        if self.pred_type == 'sequence':
            outputs = torch.cat(outputs).view(bsz, -1, self.num_classes)
            pooler_outputs = torch.cat(pooler_outputs).view(bsz, -1, self.hidden_size)

            query = self.class_weights.unsqueeze(0).expand(bsz, 
                                                        self.num_classes, 
                                                        self.hidden_size)
            weighted_output, output_mask_weights = self.output_attn(query, 
                                                                    key_value=pooler_outputs, 
                                                                    to_attend=outputs)
        else:  # token
            outputs = torch.cat(outputs).view(bsz, -1, self.num_classes, 
                                              img_dim1, img_dim2)
            pooler_outputs = torch.cat(pooler_outputs).view(bsz, -1, 
                                                            self.hidden_size, 
                                                            img_dim1, 
                                                            img_dim2)

            query = self.class_weights.unsqueeze(0) \
                .view(1, self.num_classes, self.hidden_size, -1).mean(-1) \
                .expand(bsz, self.num_classes, self.hidden_size)
            key_value = pooler_outputs.view(bsz, -1, self.hidden_size, 
                                            img_dim1 * img_dim2).mean(-1)
            weighted_output, output_mask_weights = self.output_attn(query, 
                                                                    key_value=key_value, 
                                                                    to_attend=outputs)

        
        # print('masks_weights_used', masks_weights_used.shape)
        _, predicted = torch.max(weighted_output.data, -1)
        # print('input_mask_weights', input_mask_weights.shape)
        # print('output_mask_weights', output_mask_weights.shape)
        # import pdb
        # pdb.set_trace()
        
        if self.return_tuple:
            # todo: debug for segmentation
            if self.pred_type == 'sequence':
                masks_mult = input_mask_weights.unsqueeze(2) * \
                output_mask_weights.unsqueeze(-1).unsqueeze(-1) # bsz, n_masks, n_cls, img_dim, img_dim
                masks_aggr = masks_mult.sum(1) # bsz, n_masks, img_dim, img_dim
                masks_aggr_pred_cls = masks_aggr[range(bsz), predicted].unsqueeze(1)
                max_mask_indices = output_mask_weights.max(2).indices.max(1).indices
                masks_max_pred_cls = masks_mult[range(bsz),max_mask_indices,predicted].unsqueeze(1)
            else:
                masks_aggr_pred_cls = None
                masks_max_pred_cls = None

            return AttributionOutputSOP(masks_aggr_pred_cls,
                                        weighted_output,
                                        outputs,
                                        input_mask_weights,
                                        output_mask_weights,
                                        masks_max_pred_cls)
        else:
            return weighted_output
