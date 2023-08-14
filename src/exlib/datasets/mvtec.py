import os
from glob import glob

import torch
import torch.utils.data as tud
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

#####
MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

class MVTec(Dataset):
  def __init__(self,
               data_dir,
               category,
               split = "train",
               image_size = 256, # Loads at (3,256,256) image
               good_value = 0,
               anom_value = 1,
               image_transforms = None,
               mask_transforms = None,
               download = False):
    if download:
      raise ValueError("download not implemented")

    assert category in MVTEC_CATEGORIES
    self.data_dir = data_dir
    self.images_root_dir = os.path.join(data_dir, category, split)
    self.masks_root_dir = os.path.join(data_dir, category, "ground_truth")

    assert os.path.isdir(self.images_root_dir)
    assert os.path.isdir(self.masks_root_dir)

    self.split = split
    if split == "train":
      self.image_files = sorted(glob(os.path.join(self.images_root_dir, "good", "*.png")))
    elif split == "test":
      self.image_files = sorted(glob(os.path.join(self.images_root_dir, "*", "*.png")))
    else:
      raise ValueError(f"invalid split {split} implemented")

    self.image_size = image_size

    self.image_transforms = image_transforms if image_transforms is not None else \
        transforms.Compose([
          transforms.Resize(image_size),
          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    self.mask_transforms = mask_transforms if mask_transforms is not None else \
        transforms.Compose([
          transforms.Resize(image_size),
        ])

    self.good_value = good_value
    self.anom_value = anom_value

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    image_file = self.image_files[index]
    # image_np = cv2.imread(image_file, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = torch.tensor(image_np.transpose(2,0,1))
    image = self.image_transforms(image)

    if self.split == "train":
      mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
      return image, mask, self.good_value
    else:
      if os.path.dirname(image_file).endswith("good"):
        mask = torch.zeros([1, image.shape[-2], image.shape[-1]])
        y = self.good_value
      else:
        mask_file = image_file
        sps = image_file.split("/")
        mask_file = os.path.join(self.masks_root_dir, sps[-2], sps[-1].replace(".png", "_mask.png"))
        mask_np = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = torch.tensor(mask_np != 0).byte().unsqueeze(0)
        print(f"mask shape is {mask.shape}")
        mask = self.mask_transforms(mask)
        y = self.anom_value

      return image, mask, y


