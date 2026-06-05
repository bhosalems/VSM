import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    DIGIT_WORDS = {
        '0': 'zero', '1': 'one',   '2': 'two',
        '3': 'three','4': 'four',  '5': 'five',
        '6': 'six',  '7': 'seven', '8': 'eight',
        '9': 'nine'
    }

    def __init__(self, config):
        super().__init__()
        self.split      = config.get('split')                 # e.g. 'mnist_train' or 'mnist_test'
        self.data_dir   = Path(config.get('root') + "/" + self.split)
        self.images     = sorted(self.data_dir.glob('*.png'))
        self.crop_size  = config.get('crop_size')             # e.g. 28
        self.p_uncond   = config.get('p_uncond', 0.0)         # prob of empty caption

    def __len__(self):
        return len(self.images)

    @staticmethod
    def center_crop_arr(pil_image, image_size):
        # progressively downsample by factor of 2 with BOX filter
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(
                (pil_image.size[0]//2, pil_image.size[1]//2),
                resample=Image.BOX
            )
        # resize so smallest side == image_size
        scale = image_size / min(*pil_image.size)
        new_size = tuple(round(s * scale) for s in pil_image.size)
        pil_image = pil_image.resize(new_size, resample=Image.BICUBIC)

        arr = np.array(pil_image)
        h, w = arr.shape[:2]
        top    = (h - image_size) // 2
        left   = (w - image_size) // 2
        return arr[top:top+image_size, left:left+image_size]

    def __getitem__(self, idx):
        # load and ensure grayscale->RGB for consistency if needed
        img = Image.open(self.images[idx]).convert('RGB')
        # center‐crop/rescale to square
        arr = self.center_crop_arr(img, self.crop_size)
        # normalize to [-1,1]
        img_tensor = (arr.astype(np.float32) / 127.5) - 1.0

        # parse label from filename, e.g. '3_12345.png' → 'three'
        digit = self.images[idx].stem.split('_', 1)[0]
        caption = self.DIGIT_WORDS.get(digit, "")
        # unconditioned caption with prob p_uncond
        if np.random.rand() < self.p_uncond:
            caption = ""

        return {
            'image': img_tensor,    # shape: (H, W, 3), dtype=float32
            'caption': caption
        }