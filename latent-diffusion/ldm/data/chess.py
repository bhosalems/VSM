from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os
import numpy as np
import json
import glob

class ChessDatasetUncond(Dataset):
    def __init__(self, config):
        super().__init__()
        self.split = config.get('split')
        self.data_dir = Path(config.get('root'))
        self.image_dir = os.path.join(self.data_dir, self.split+"_images")
        self.crop_size = config.get('crop_size')
        self.inference = config.get('inference')
        self.images = glob.glob(os.path.join(self.image_dir, "*.png"))
   
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def resize_with_aspect_ratio(tile, target_size):
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
        img = Image.open(os.path.join(self.image_dir, self.images[idx]))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.resize_with_aspect_ratio(img, self.crop_size)
        img = np.array(img)
        img = (img / 127.5 - 1).astype(np.float32)
        return {'image' : img}

class ChessDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.split = config.get('split')
        self.data_dir = Path(config.get('root'))
        self.image_dir = os.path.join(self.data_dir, self.split+"_images")
        self.crop_size = config.get('crop_size')
        self.inference = config.get('inference')
        self.p_uncond = config.get("p_uncond", 0)
        self.fen = json.load(open(os.path.join(config.get('root'), self.split+"_fen.json")))
        self.fen_list = list(self.fen.items())  # Convert to list for indexing
    def __len__(self):
        return len(self.fen_list)
    
    @staticmethod
    def resize_with_aspect_ratio(tile, target_size):
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
        caption_fen = self.fen_list[idx][1].split(" ")[0]
        img = Image.open(os.path.join(self.image_dir, self.fen_list[idx][0]+".png"))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.resize_with_aspect_ratio(img, self.crop_size)
        img = np.array(img)
        img = (img / 127.5 - 1).astype(np.float32)
        
        if np.random.rand() < self.p_uncond:
            caption_fen = ""
        return {'image' : img, "caption": caption_fen}