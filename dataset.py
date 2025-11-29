# dataset.py 

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class InvoiceSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

        self.mask_map = {}
        for f in os.listdir(masks_dir):
            if f.lower().endswith(image_extensions):
                name = os.path.splitext(f)[0]
                self.mask_map[name.lower()] = os.path.join(masks_dir, f)

        self.pairs = []
        for img_name in all_images:
            base = os.path.splitext(img_name)[0].lower()
            found = False

            if base in self.mask_map:
                self.pairs.append((img_name, self.mask_map[base]))
                found = True
            else:
                if base.isdigit():
                    num = int(base)
                    for pad in [2, 3, 4]:
                        padded = f"{num:0{pad}d}".lower()
                        if padded in self.mask_map:
                            self.pairs.append((img_name, self.mask_map[padded]))
                            found = True
                            break
                if not found and base.lstrip("0") in self.mask_map:
                    clean = base.lstrip("0") or "0"
                    if clean in self.mask_map:
                        self.pairs.append((img_name, self.mask_map[clean]))
                        found = True

            if not found:
                print(f"警告：找不到 mask → {img_name}")

        print(f"成功載入 {len(self.pairs)} 對圖片-mask 配對")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_path = self.pairs[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 讀圖片
        image = Image.open(img_path).convert("RGB")

        # 讀彩色 mask
        mask = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask)

        # RGB → class id
        label = np.zeros(mask_np.shape[:2], dtype=np.int64)
        label[np.all(mask_np == [255, 0,   0], axis=-1)] = 1   # 發票號碼
        label[np.all(mask_np == [  0, 255, 0], axis=-1)] = 2   # 日期
        label[np.all(mask_np == [  0,   0, 255], axis=-1)] = 3   # 總金額

        # 轉換圖片
        if self.transform:
            image = self.transform(image)

        # 轉換 mask
        if self.mask_transform:
            label_pil = Image.fromarray(label.astype(np.uint8))
            label = self.mask_transform(label_pil)           # 這裡會變成 (1, H, W) 的 tensor
            label = label.squeeze(0)                         # 變成 (H, W)
        else:
            label = torch.from_numpy(label).long()

        return image, label
