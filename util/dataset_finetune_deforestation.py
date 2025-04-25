import warnings
from rasterio import logging

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)


CATEGORIES = ["plantation", "logging", "mining", "grassland_shrubland"]


class SatelliteDataset(Dataset):
    """
    Abstract class.
    """
    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Builds train/eval data transforms for the dataset class.
        :param is_train: Whether to yield train or eval data transform/augmentation.
        :param input_size: Image input size (assumed square image).
        :param mean: Per-channel pixel mean value, shape (c,) for c channels
        :param std: Per-channel pixel std. value, shape (c,)
        :return: Torch data transform for the input image before passing to model
        """
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD

        # train transform
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.3, 1.0), interpolation=interpol_mode),  # 3 is bicubic
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # eval transform
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

        # t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


#########################################################
# Deforestation (Sentinel)
#########################################################
import os
from glob import glob
import torch
from torch.utils.data import Dataset, random_split
import rasterio
import numpy as np

class DeforestationDataset(SatelliteDataset):
    """
    File-based per-pixel segmentation dataset.
    Lazily loads one image + mask pair from disk per __getitem__.
    """
    def __init__(self, image_paths, mask_paths):
        # in_c should match the number of bands in your TIFFs (e.g. 12)
        super().__init__(in_c=12)
        assert len(image_paths) == len(mask_paths)
        self.images = image_paths
        self.masks  = mask_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image (C, H, W)
        with rasterio.open(self.images[idx]) as src:
            img = src.read().astype(np.float32)
        # load mask (H, W)
        with rasterio.open(self.masks[idx]) as src:
            mask = src.read(1).astype(np.int64)

        # return as dict to match your training loop
        return { 'image': torch.from_numpy(img), 
                 'mask' : torch.from_numpy(mask) }

def build_deforestation_datasets(
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    Uses get_processed_data() (which returns a TensorDataset of ALL your
    (image,mask) pairs), splits it into train+val subsets, and returns
    two small Dataset wrappers that yield {'image', 'mask'} dicts.
    """
    # 1) load the full TensorDataset of (img, mask)
    full_ds = get_processed_data()   # <-- unchanged!

    # 2) split lengths
    total = len(full_ds)
    train_n = int(total * train_ratio)
    val_n   = total - train_n

    # 3) random split with a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_tds, val_tds = random_split(full_ds, [train_n, val_n], generator=generator)

    # 4) wrap each Subset in your simple DeforestationDataset so it returns dicts
    train_ds = DeforestationDataset(train_tds)
    val_ds   = DeforestationDataset(val_tds)

    return train_ds, val_ds