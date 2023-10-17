import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
from collections import namedtuple
from typing import Optional, Union, List

import segmentation_models_pytorch as smp
from ..modules.sop import SOPImageSeg, SOPConfig, get_chained_attr



# The kinds of splits we can do
SPLIT_TYPES = ["train", "test", "train_video", "test_video"]

# Splitting images by video source
VIDEO_GLOBS = \
      [f"AdnanSet_LC_{i}_*" for i in range(1,165)] \
    + [f"AminSet_LC_{i}_*" for i in range(1,11)] \
    + [f"cholec80_video0{i}_*" for i in range(1,10)] \
    + [f"cholec80_video{i}_*" for i in range(10,81)] \
    + ["HokkaidoSet_LC_1_*", "HokkaidoSet_LC_2_*"] \
    + [f"M2CCAI2016_video{i}_*" for i in range(81,122)] \
    + [f"UTSWSet_Case_{i}_*" for i in range(1,13)] \
    + [f"WashUSet_LC_01_*"]

#
class AbdomenOrgans(Dataset):
    def __init__(self,
                 data_dir,
                 images_dirname = "images",
                 gonogo_labels_dirname = "gonogo_labels",
                 organ_labels_dirname = "organ_labels",
                 split = "train",
                 train_ratio = 0.8,
                 image_height = 384,  # Default image height / widths
                 image_width = 640,
                 image_transforms = None,
                 label_transforms = None,
                 split_seed = 1234,
                 download = False):
        if download:
            raise ValueError("download not implemented")

        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, images_dirname)
        self.gonogo_labels_dir = os.path.join(data_dir, gonogo_labels_dirname)
        self.organ_labels_dir = os.path.join(data_dir, organ_labels_dirname)

        assert os.path.isdir(self.images_dir)
        assert os.path.isdir(self.gonogo_labels_dir)
        assert os.path.isdir(self.organ_labels_dir)
        assert split in SPLIT_TYPES
        self.split = split

        # Split not regarding video
        torch.manual_seed(split_seed)
        if split == "train" or split == "test":
            all_image_filenames = sorted(os.listdir(self.images_dir))
            num_all, num_train = len(all_image_filenames), int(len(all_image_filenames) * train_ratio)
            idx_perms = torch.randperm(num_all)
            todo_idxs = idx_perms[:num_train] if split == "train" else idx_perms[num_train:]
            self.image_filenames = sorted([all_image_filenames[i] for i in todo_idxs])

        # Split by the video source
        elif split == "train_video" or split == "test_video":
            num_all, num_train = len(VIDEO_GLOBS), int(len(VIDEO_GLOBS) * train_ratio)
            idx_perms = torch.randperm(num_all)
            todo_idxs = idx_perms[:num_train] if "train" in split else idx_perms[num_train:]

            image_filenames = []
            for idx in todo_idxs:
                image_filenames += glob.glob(os.path.join(self.images_dir, VIDEO_GLOBS[idx]))
            self.image_filenames = sorted(image_filenames)

        else:
            raise NotImplementedError()

        self.image_height = image_height
        self.image_width = image_width

        # Image transforms
        if image_transforms is None:
            if "train" in split:
                self._image_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_height, image_width), antialias=True),
                    transforms.RandomRotation(60),
                ])

                self.image_transforms = lambda image, seed: \
                        (torch.manual_seed(seed), self._image_transforms(image))[1]
            else:
                self.image_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_height, image_width), antialias=True)
                ])
        else:
            assert callable(image_transforms)
            self.image_transforms = image_transforms

        # Label transforms
        if label_transforms is None:
            if "train" in split:
                self._label_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((image_height, image_width), antialias=True),
                    transforms.RandomRotation(60),
                ])

                self.label_transforms = lambda label, seed: \
                        (torch.manual_seed(seed), self._label_transforms(label))[1]

            else:
                self.label_transforms = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Resize((image_height, image_width), antialias=True)
                ])
        else:
            assert callable(label_transforms)
            self.label_transforms = label_transforms

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_file = os.path.join(self.images_dir, self.image_filenames[idx])
        organ_label_file = os.path.join(self.organ_labels_dir, self.image_filenames[idx])
        gonogo_label_file = os.path.join(self.gonogo_labels_dir, self.image_filenames[idx])

        # Read image and label
        image = Image.open(image_file).convert("RGB")
        organ_label = Image.open(organ_label_file).convert("L") # L is grayscale
        gonogo_label = Image.open(gonogo_label_file).convert("L")

        if self.split == "train":
            seed = torch.seed()
            image = self.image_transforms(image, seed)
            organ_label = self.label_transforms(organ_label, seed)
            gonogo_label = self.label_transforms(gonogo_label, seed)
        else:
            image = self.image_transforms(image)
            organ_label = self.label_transforms(organ_label)
            gonogo_label = self.label_transforms(gonogo_label)

        organ_label = (organ_label * 255).round().long()
        gonogo_label = (gonogo_label * 255).round().long()
        return image, organ_label, gonogo_label



# Basic classification model
class AbdomenClsModel(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1),   # (N,256,32,32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1), # (N,128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # (N,64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),   # (N,32,4,4)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(32*4*4, num_classes)
        )

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == 3
        x = F.interpolate(x, size=[64,64])
        y = self.layers(x)
        return y


# Basic segmentation model
class AbdomenSegModel(nn.Module):
    def __init__(self, num_segments, in_channels=3,
                 encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.unet = smp.Unet(encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             in_channels=in_channels,
                             classes=num_segments)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % 32 == 0 and W % 32 == 0
        return self.unet(x)


ModelOutput = namedtuple("ModelOutput", ["logits", "pooler_output"])

def convert_idx_masks_to_bool(masks, num_masks=None):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    if num_masks is not None:
        unique_idxs = torch.arange(num_masks).to(masks.device)
    else:
        unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[-2], 
                                     masks.shape[-1])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool


class Unet(smp.Unet):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__(encoder_name=encoder_name,
                         encoder_depth=encoder_depth,
                         encoder_weights=encoder_weights,
                         decoder_use_batchnorm=decoder_use_batchnorm,
                         decoder_channels=decoder_channels,
                         decoder_attention_type=decoder_attention_type,
                         in_channels=in_channels,
                         classes=classes,
                         activation=activation,
                         aux_params=aux_params)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return ModelOutput(logits=masks,
                           pooler_output=decoder_output)


class OrganGNGModel(nn.Module):
    def __init__(self, num_organs=4):
        super(OrganGNGModel, self).__init__()
        self.num_organs = num_organs
        self.organ_seg_model = None
        self.gonogo_model = None
    
    def restore_organ(self, restore_organ=None):
        if restore_organ:
            state_dict = torch.load(restore_organ)
            self.organ_seg_model.load_state_dict(state_dict['model_state_dict'], 
                                                strict=False)

    def restore_gonogo(self, restore_gonogo=None):
        if restore_gonogo:
            state_dict = torch.load(restore_gonogo)
            self.gonogo_model.load_state_dict(state_dict['model_state_dict'], 
                                              strict=False)


class OrganGNGSopSelectorModel(OrganGNGModel):
    def __init__(self, num_organs=4):
        super().__init__(num_organs=num_organs)
        self.organ_seg_model = Unet(
                encoder_name = "resnet50",
                encoder_weights = "imagenet",
                in_channels = 3,
                classes = num_organs,
                activation = "softmax2d")

        self.gonogo_model = Unet(
                encoder_name = "resnet50",
                encoder_weights = "imagenet",
                in_channels = 3,
                classes = 3,
                activation = "softmax2d")
        
        config = SOPConfig(
            attn_patch_size=32,
            num_heads=1,
            num_masks_sample=20,
            finetune_layers=['segmentation_head'],
            hidden_size=16,
            num_labels=3,
            image_size=(352, 640),
            num_channels=num_organs,
            num_masks_max=100
        )

        # allow specifying args to be different from in the json file
        class_weights = get_chained_attr(self.gonogo_model, config.finetune_layers[0])[0].weight.view(3,-1).clone()
        self.sop_model = SOPImageSeg(config, 
                              self.gonogo_model,
                              class_weights =class_weights
                                    )
    
    def freeze_organ(self):
        for name, param in self.organ_seg_model.named_parameters():
            param.requires_grad = False

    def forward(self, x, get_organs=False, return_dict=False):
        organ_output = self.organ_seg_model(x)
        organ_mask_output = organ_output.logits.argmax(dim=1).byte()
        # organ_mask_output_resize = F.interpolate(organ_mask_output, size=(x.shape[2], x.shape[3]), mode='nearest')
        organ_mask_output_bools = []
        for organ_mask_output_i in organ_mask_output:
            organ_mask_output_bool = convert_idx_masks_to_bool(organ_mask_output_i, 
                                                               num_masks=self.num_organs).int()
            organ_mask_output_bools.append(organ_mask_output_bool)
        organ_mask_output_bools = torch.stack(organ_mask_output_bools)
        gonogo_output = self.sop_model(x, input_mask_weights=organ_mask_output_bools, epoch=0,
                                       mask_batch_size=8, return_tuple=True)
        
        if return_dict:
            return {
                'organ_output': organ_output,
                'gonogo_output': gonogo_output
            }

        if get_organs:
            return organ_output.logits, gonogo_output.logits
        return gonogo_output.logits