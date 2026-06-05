from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np

class CatsDogsDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.split = config.get('split')
        self.data_dir = Path(os.path.join(config.get('root'), self.split))
        self.images = list(self.data_dir.glob('*.jpg'))
        self.crop_size = config.get('crop_size')
        self.inference = config.get('inference')
        self.p_uncond = config.get("p_uncond", 0)
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def resize_with_aspect_ratio(tile, target_size): # Just increases the size if needed
        w, h = tile.size
        if h < target_size or w < target_size:
            aspect_ratio = w / h
            if h < w:
                new_h = target_size
                new_w = int(target_size * aspect_ratio)
            else:
                new_w = target_size
                new_h = int(target_size / aspect_ratio)
            tile = tile.resize((new_w, new_h), Image.BICUBIC)
        return tile
    
    @staticmethod
    def center_crop_arr(pil_image, image_size):
        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - image_size) // 2
        crop_x = (arr.shape[1] - image_size) // 2
        return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img = np.array(self.resize_with_aspect_ratio(img, self.crop_size), dtype=np.float32)
        img = self.center_crop_arr(img, self.crop_size)
        img = (img / 127.5 - 1).astype(np.float32)
        # if np.random.rand() < 0.5 and not self.inference:
        #     img = np.flip(img, axis=1).copy()
        # if np.random.rand() < 0.5 and not self.inference:
        #     img = np.flip(img, axis=0).copy()
        
        caption = ''
        if np.random.rand() < self.p_uncond:
            caption = ""
        return {'image' : img, "caption": caption}