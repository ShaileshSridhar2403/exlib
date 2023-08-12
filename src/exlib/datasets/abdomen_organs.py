import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import glob

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
               gonogo_masks_dirname = "gonogo_masks",
               organ_masks_dirname = "organ_masks",
               split = "train",
               train_ratio = 0.8,
               image_height = 360,  # Default image height / widths
               image_width = 640,
               image_transforms = None,
               mask_transforms = None,
               split_seed = 1234,
               download = False):
    if download:
      raise ValueError("download not implemented")

    self.data_dir = data_dir
    self.images_dir = os.path.join(data_dir, images_dirname)
    self.gonogo_masks_dir = os.path.join(data_dir, gonogo_masks_dirname)
    self.organ_masks_dir = os.path.join(data_dir, organ_masks_dirname)

    assert os.path.isdir(self.images_dir)
    assert os.path.isdir(self.gonogo_masks_dir)
    assert os.path.isdir(self.organ_masks_dir)
    assert split in SPLIT_TYPES
    self.split = split

    # Split not regarding video
    torch.manual_seed(split_seed)
    if split == "train" or split == "test":
      all_image_names = sorted(os.listdir(self.images_dir))
      num_all, num_train = len(all_image_names), int(len(all_image_names) * train_ratio)
      idx_perms = torch.randperm(num_all)
      todo_idxs = idx_perms[:num_train] if split == "train" else idx_perms[num_train:]
      self.image_names = sorted([all_image_names[i] for i in todo_idxs])

    # Split by the video source
    elif split == "train_video" or split == "test_video":
      num_all, num_train = len(VIDEO_GLOBS), int(len(VIDEO_GLOBS) * train_ratio)
      idx_perms = torch.randperm(num_all)
      todo_idxs = idx_perms[:num_train] if "train" in split else idx_perms[num_train:]

      image_names = []
      for idx in todo_idxs:
        image_names += glob.glob(os.path.join(self.images_dir, VIDEO_GLOBS[idx]))
      self.image_names = sorted(image_names)

    else:
      raise NotImplementedError()

    self.image_height = image_height
    self.image_width = image_width

    # The preprocessing pipeline is different based on the split form
    if "train" in split:
      self._image_transforms = transforms.Compose([
        transforms.Resize((image_height, image_width), antialias=False),
        transforms.RandomRotation(60), 
      ])

      self._mask_transforms = transforms.Compose([
        transforms.Resize((image_height, image_width), antialias=False),
        transforms.RandomRotation(60), 
      ])

      self.image_transforms = image_transforms if image_transforms is not None else \
          lambda image, seed : (torch.manual_seed(seed), self._image_transforms(image))[1]

      self.mask_transforms = mask_transforms if mask_transforms is not None else \
          lambda mask, seed : (torch.manual_seed(seed), self._mask_transforms(mask))[1]

    elif "test" in split:
      self.image_transforms = image_transforms if image_transforms is not None else \
          transforms.Resize((image_height, image_width))

      self.mask_transforms = mask_transforms if mask_transforms is not None else \
          transforms.Resize((image_height, image_width))

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, idx):
    image_file = os.path.join(self.images_dir, self.image_names[idx])
    organ_mask_file = os.path.join(self.organ_masks_dir, self.image_names[idx])
    gonogo_mask_file = os.path.join(self.gonogo_masks_dir, self.image_names[idx])

    # Read image and mask
    image_np = cv2.imread(image_file)
    organ_mask_np = cv2.imread(organ_mask_file, cv2.IMREAD_GRAYSCALE)
    gonogo_mask_np = cv2.imread(gonogo_mask_file, cv2.IMREAD_GRAYSCALE)

    image = torch.tensor(image_np.transpose(2,0,1))                     # (3, H, C)
    organ_mask = torch.tensor(organ_mask_np).unsqueeze(dim=0).byte()    # (1, H, C)
    gonogo_mask = torch.tensor(gonogo_mask_np).unsqueeze(dim=0).byte()  # (1, H, C)

    if self.split == "train":
      seed = torch.seed()
      image = self.image_transforms(image, seed)
      organ_mask = self.mask_transforms(organ_mask, seed)
      gonogo_mask = self.mask_transforms(gonogo_mask, seed)
    else:
      image = self.image_transforms(image)
      organ_mask = self.mask_transforms(organ_mask)
      gonogo_mask = self.mask_transforms(gonogo_mask)

    return image, organ_mask, gonogo_mask


