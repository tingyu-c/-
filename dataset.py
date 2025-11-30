# dataset.py 

import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class InvoiceDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith('.png')])

        assert len(self.images) == len(self.masks), "Images / Masks count mismatch!"

        # RGB → class id
        self.color_to_class = {
            (0, 0, 0): 0,
            (255, 0, 0): 1,
            (0, 255, 0): 2,
            (0, 0, 255): 3
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # --- 讀圖片 ---
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        # --- 讀 mask（RGB） ---
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask_img = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask_img)

        # --- RGB mask → class mask (H, W) ---
        class_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)

        for rgb, cid in self.color_to_class.items():
            match = np.all(mask_np == np.array(rgb), axis=-1)
            class_mask[match] = cid

        class_mask = Image.fromarray(class_mask)  # 回 PIL（才能做 Resize）

        # --- apply transforms ---
        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            class_mask = self.mask_transform(class_mask)

        # --- 最重要：轉成 LongTensor（CrossEntropyLoss 需要） ---
        class_mask = torch.from_numpy(np.array(class_mask)).long()  # (H, W)

        return image, class_mask

