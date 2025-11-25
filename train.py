import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# 確保這兩個檔案在同一目錄下
from dataset import InvoiceSegDataset
from unet_model import UNet

import torchvision.transforms as T

# ----------------------------
# 輔助函式：將正規化的圖片轉回 PIL 圖片
# ----------------------------
def visualize_epoch(img, true_mask, pred_mask, save_prefix):
    """
    輸出訓練可視化：
    - img: 輸入圖片 (Tensor)
    - true_mask: 真實遮罩 (Tensor, 值域 0~3)
    - pred_mask: 預測遮罩 (Tensor, 值域 0~3)
    """
    os.makedirs("visualize", exist_ok=True)

    # ImageNet 標準化參數 (用於反正規化)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # 反正規化
    inv_normalize = T.Normalize(
        mean=[-m/s for m, s in zip(MEAN, STD)],
        std=[1/s for s in STD]
    )
    img = inv_normalize(img.clone())
    
    # 將圖片從 Tensor 轉為 numpy (H, W, C) 並轉換為 0-255 整數
    img_np = (img.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(f"visualize/{save_prefix}_img.png")

    # 遮罩可視化：將 0~3 的類別 ID 映射到 0~255 的灰度值
    # 類別 0=0, 1=85, 2=170, 3=255
    color_scale = 255 // 3 
    
    true_mask_vis = (true_mask.numpy() * color_scale).astype(np.uint8)
    Image.fromarray(true_mask_vis).save(
        f"visualize/{save_prefix}_true_mask.png"
    )

    pred_mask_vis = (pred_mask.numpy() * color_scale).astype(np.uint8)
    Image.fromarray(pred_mask_vis).save(
        f"visualize/{save_prefix}_pred_mask.png"
    )


# ----------------------------
# 主程式
# ----------------------------
def main():
    images_dir = "data/images"
    masks_dir = "masks"
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 🚨 修正: 類別數必須與 inference.py 一致 (0=背景, 1=號碼, 2=日期, 3=金額)
    N_CLASSES = 4 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")

    # 數據增強 (Data Augmentation)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 標準化
    ])
    
    # 遮罩處理：只需要 Resize，不需要 ToTensor 或 Normalize
    mask_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
    ])

    dataset = InvoiceSegDataset(images_dir, masks_dir, transform, mask_transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    if len(dataset) == 0:
        print("🚨 錯誤: 資料集為空。請確認 'data/images' 和 'masks' 資料夾中有對應的圖片和遮罩檔案。")
        return

    # 🚨 修正: 類別數改為 N_CLASSES=4
    model = UNet(n_channels=3, n_classes=N_CLASSES).to(device)

    # 🚨 修正: 損失函式改為 CrossEntropyLoss (用於多類別分割)
    # 遮罩 (masks) 必須是 LongTensor 且 shape 為 (N, H, W)
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    current_best_loss = float('inf')

    print("\n開始訓練...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        print(f"\n==== Epoch {epoch}/{epochs} ====\n")

        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device)
            # 遮罩 (masks) 已經是 LongTensor 且維度正確 (N, H, W)
            masks = masks.to(device) 

            preds = model(imgs) # preds shape: (N, C, H, W)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            print(f"[{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")

            # --- 可視化第一張 sample ---
            if batch_idx == 0:
                # 預測結果取 argmax (從 C 維度中選擇機率最高的類別)
                pred_mask = torch.argmax(preds[0], dim=0).cpu() # (H, W)
                # 輸出到 visualize/ 目錄 (masks[0] 是 (H, W))
                visualize_epoch(imgs[0].cpu(), masks[0].cpu(), pred_mask,
                                f"epoch{epoch}")

        # 儲存 Checkpoint
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} 訓練完成. Avg Loss: {avg_loss:.4f}")
        
        # 儲存當前 Checkpoint
        current_ckpt_path = os.path.join(checkpoint_dir, f"unet_epoch{epoch}.pth")
        torch.save(model.state_dict(), current_ckpt_path)

        # 🚨 修正: 為了讓 Streamlit 應用程式始終載入最新的模型，我們只保留一個檔案。
        # 刪除前一個 Checkpoint
        if epoch > 1:
             prev_ckpt_path = os.path.join(checkpoint_dir, f"unet_epoch{epoch-1}.pth")
             if os.path.exists(prev_ckpt_path):
                os.remove(prev_ckpt_path)
                print(f"已刪除舊 Checkpoint: {os.path.basename(prev_ckpt_path)}")
        
        # 讓 Streamlit 應用程式載入這個檔案
        # 這裡可以選擇將最好的模型另外儲存 (Best Model)
        if avg_loss < current_best_loss:
            current_best_loss = avg_loss
            best_ckpt_path = os.path.join(checkpoint_dir, "best_unet_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"🎉 新的最佳模型已儲存: {os.path.basename(best_ckpt_path)}")


if __name__ == "__main__":
    main()