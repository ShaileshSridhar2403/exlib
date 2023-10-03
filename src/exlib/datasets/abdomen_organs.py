import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob

import segmentation_models_pytorch as smp

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



class AbodmenModel(nn.Module):
    def __init__(self, in_channels, out_channels,
                 encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = smp.Unet(encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             in_channels=in_channels,
                             classes=out_channels)

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % 32 == 0 and W % 32 == 0
        return self.unet(x)


