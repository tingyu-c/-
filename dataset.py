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

        # 1. 讀取所有圖片檔名
        all_image_names = [f for f in os.listdir(images_dir)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        # 2. 建立一個遮罩檔案的查找表 (Map)
        # 鍵: 基礎名稱 (e.g., '1', '126', '001')
        # 值: 完整路徑 (e.g., 'masks/001.png')
        mask_map = {}
        for mask_name in os.listdir(masks_dir):
            if mask_name.lower().endswith((".jpg", ".jpeg", ".png")):
                base_name = mask_name.rsplit(".", 1)[0]
                mask_map[base_name] = os.path.join(masks_dir, mask_name)

        # 3. 找出有效的圖片-遮罩配對
        self.images_to_load = []
        for img_name in all_image_names:
            base_name = img_name.rsplit(".", 1)[0]
            
            # --- 嘗試幾種常見的命名配對邏輯 ---
            
            # 嘗試 1: 圖片基礎名 (e.g., '126')
            if base_name in mask_map:
                self.images_to_load.append({
                    'img_name': img_name,
                    'mask_path': mask_map[base_name]
                })
                continue
                
            # 嘗試 2: 如果圖片名稱是數字，嘗試補零配對 (e.g., '1' -> '001', '01')
            if base_name.isdigit():
                num = int(base_name)
                # 嘗試 001, 01, 0001
                for padding in [2, 3, 4]: 
                    padded_name = f"{num:0{padding}d}"
                    if padded_name in mask_map:
                        self.images_to_load.append({
                            'img_name': img_name,
                            'mask_path': mask_map[padded_name]
                        })
                        break # 找到後跳出 padding 迴圈
                else:
                    # 如果內層 for 迴圈沒有 break (表示沒找到配對)，則繼續下一個圖片
                    continue
            
            # 嘗試 3: 如果遮罩名稱是數字，嘗試圖片名稱不補零配對
            # (已經在嘗試 1, 2 中涵蓋了)
            
        print(f"📌 資料集載入：{len(self.images_to_load)} 張圖片")
        if len(self.images_to_load) == 0:
             print("⚠️ 警告：沒有找到任何配對的圖片和遮罩檔案。請檢查 'data/images' 和 'masks' 資料夾的檔案名稱是否一致或有補零差異。")


    def __len__(self):
        return len(self.images_to_load)

    def __getitem__(self, idx):
        item = self.images_to_load[idx]
        img_name = item['img_name']
        mask_path = item['mask_path']

        # load image
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # 載入遮罩並轉為灰度圖 (L)
        # mask_path 現在已經是正確的完整路徑
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        
        # 遮罩量化：將 0-255 的灰度值量化為 0, 1, 2, 3 四個類別 ID
        # 假設最大值是 255，我們除以 (255/3) 來量化，四捨五入到最近的整數
        mask_np = np.round(mask_np / (255 / 3.0)).astype(np.int64)

        # 確保值在 [0, N_CLASSES-1] 範圍內
        mask_np = np.clip(mask_np, 0, 3) # 4 個類別: 0, 1, 2, 3 (UNet 的輸出類別數應為 4)

        if self.transform:
            img = self.transform(img)

        # 遮罩轉為 LongTensor
        if self.mask_transform:
            # 必須使用 PIL Image 才能應用 Resize 或其他變換
            mask_img = self.mask_transform(Image.fromarray(mask_np, mode='L'))
            # 轉換為 LongTensor (CrossEntropyLoss 要求的格式)
            mask_tensor = torch.as_tensor(np.array(mask_img), dtype=torch.long)
        else:
            mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)
            
        return img, mask_tensor