import os
from typing import OrderedDict
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from captionizer import caption_from_path, generic_captions_from_path
from captionizer import find_images

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root,
        set,
        repeats,
        resolution,
        resampler,
        center_crop,
        mirror_prob,
        reg=False,
        placeholder_token="rock",
        coarse_class_text=None,
        mixing_prob=0.25,
        per_image_tokens=False,
        token_only=False
    ):
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.coarse_class_text = coarse_class_text
        self.resolution = resolution
        self.resampler = {
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[resampler]
        
        if per_image_tokens:
            assert self.num_images < len(
                per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = int(self.num_images * repeats)

        self.flip = transforms.RandomHorizontalFlip(p=mirror_prob)
        self.reg = reg
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

    def __len__(self):
        return self._length

    def __getitem__(self                                                                                                                                            , i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        example["caption"] = ""
        if self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        H, W, = img.shape[0], img.shape[1]
        max = min(H, W)
        if self.center_crop and not H == W:
            img = img[
                (h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)
        
        if self.resolution is not None and not self.resolution == max:
            image = image.resize(
                (self.resolution, self.resolution),
                resample=self.resampler,
                reducing_gap=3)
            image = ImageEnhance.Sharpness(image).enhance(1.05)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        
        return example
