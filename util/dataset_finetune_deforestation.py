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
import torch
from torch.utils.data import Dataset, random_split
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from utils.preprocessing import get_processed_data

class DeforestationDataset(SatelliteDataset):
    """
    Wraps a TensorDataset of (img_tensor, mask_tensor) tuples
    and simply returns them—no extra transform needed.
    """
    def __init__(self, tensor_dataset: torch.utils.data.TensorDataset):
        # in_c should match number of channels in your processed images (e.g. 12)
        super().__init__(in_c=12)
        self.ds = tensor_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        return {
            'image': img,
            'mask' : mask
        }

def build_deforestation_datasets(
    train_ratio: float = 0.8,
    seed: float = 42
):
    """
    Splits the single TensorDataset from get_processed_data() into train/val
    and wraps each in our DeforestationDataset.
    """
    # 1) Get the full TensorDataset of (img, mask)
    full_ds = get_processed_data(subset=True)

    # 2) Compute lengths & split
    total = len(full_ds)
    train_len = int(total * train_ratio)
    val_len   = total - train_len
    g = torch.Generator().manual_seed(seed)
    train_tds, val_tds = random_split(full_ds, [train_len, val_len], generator=g)

    # 3) Wrap in our simple Dataset
    train_ds = DeforestationDataset(train_tds)
    val_ds   = DeforestationDataset(val_tds)

    return train_ds, val_ds

