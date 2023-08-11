import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

class AbdomenGNG(Dataset):
  def __init__(self,
               data_dir,
               image_train_dirname = "images_train",
               image_test_dirname = "images_test",
               mask_dirname = "masks",
               split = "train",
               image_height = 352,  # Default image height / widths
               image_width = 640,
               image_transforms = None,
               mask_transforms = None,
               download = False):
    if download:
      raise ValueError("download not implemented")

    self.split = split
    self.data_dir = data_dir
    self.mask_dir = os.path.join(data_dir, mask_dirname)
    if split == "train":
      self.image_dir = os.path.join(data_dir, image_train_dirname)
    else:
      self.image_dir = os.path.join(data_dir, image_test_dirname)

    assert os.path.isdir(self.mask_dir), f"mask directory does not exist {self.mask_dir}"
    assert os.path.isdir(self.image_dir), f"image directory does not exist {self.image_dir}"

    self.image_names = sorted(os.listdir(self.image_dir))

    self.image_height = image_height
    self.image_width = image_width
    if split == "train":
      self._image_transforms = transforms.Compose([
        transforms.Resize((image_height, image_width), antialias=False),
        transforms.RandomRotation(60), 
        # transforms.RandomHorizontalFlip(p=0.4), # TODO: ask doctors if flips make sense?
        # transforms.RandomVerticalFlip(p=0.4),
      ])
      self.image_transforms = \
          lambda image, seed : (torch.manual_seed(seed), self._image_transforms(image))[1]

      self._mask_transforms = transforms.Compose([
        transforms.Resize((image_height, image_width), antialias=False),
        transforms.RandomRotation(60), 
        # transforms.RandomHorizontalFlip(p=0.4),
        # transforms.RandomVerticalFlip(p=0.4),
      ])
      self.mask_transforms = \
          lambda mask, seed : (torch.manual_seed(seed), self._mask_transforms(mask))[1]

    else:
      self.image_transforms = transforms.Resize((image_height, image_width))
      self.mask_transforms = transforms.Resize((image_height, image_width))
    
    folder = os.path.join

  def __len__(self):
    return len(self.image_names)

  def __getitem__(self, index):
    image_file = os.path.join(self.image_dir, self.image_names[index])
    mask_file = os.path.join(self.mask_dir, self.image_names[index])

    # Read image and mask
    np_image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB) # BGR to RGB
    np_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    image = torch.tensor(np_image.transpose(2,0,1))       # (3, H, C)
    mask = torch.tensor(np_mask).unsqueeze(dim=0).byte()  # (1, H, C)

    if self.split == "train":
      seed = torch.seed()
      image = self.image_transforms(image, seed)
      mask = self.mask_transforms(mask, seed)
    else:
      image = self.image_transforms(image)
      mask = self.mask_transforms(mask)

    return image, mask


