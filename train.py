import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T

from dataset import InvoiceSegDataset
from unet_model import UNet


# ---------------------------- 可視化函式（彩色版）----------------------------
def visualize_epoch(img, true_mask, pred_mask, save_prefix):
    os.makedirs("visualize", exist_ok=True)

    # 反正規化 ImageNet
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img = inv_normalize(img.clone())
    img = torch.clamp(img, 0, 1)
    img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(img_np).save(f"visualize/{save_prefix}_img.png")

    # 彩色遮罩（紅=invoice_no, 綠=date, 藍=total_amount）
    def mask_to_color(mask_tensor):
        mask = mask_tensor.cpu().numpy()
        color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        color[mask == 1] = [255, 0, 0]    # 紅色
        color[mask == 2] = [0, 255, 0]    # 綠色
        color[mask == 3] = [0, 0, 255]    # 藍色
        return color

    true_color = mask_to_color(true_mask)
    pred_color = mask_to_color(pred_mask)

    Image.fromarray(true_color).save(f"visualize/{save_prefix}_true.png")
    Image.fromarray(pred_color).save(f"visualize/{save_prefix}_pred.png")


# ---------------------------- 主程式 ----------------------------
def main():
    images_dir = "images"
    masks_dir   = "masks"          # 這裡一定要是放「彩色 mask」的地方！
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置: {device}")

    # 跟 inference.py 完全一致：512x512
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = T.Compose([
        T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),                                      # 加上這行
    ])

    dataset = InvoiceSegDataset(images_dir, masks_dir, transform, mask_transform)
    loader  = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    print(f"載入 {len(dataset)} 張訓練資料")

    if len(dataset) == 0:
        print("資料集為空！請先執行 json_to_mask.py 產生彩色 mask")
        return

    model = UNet(n_channels=3, n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # 更穩

    epochs = 50
    best_loss = float('inf')

    print("開始訓練！每 epoch 會存 visualize 圖片在 visualize/ 資料夾\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        for i, (imgs, masks) in enumerate(loader):
            imgs  = imgs.to(device)
            masks = masks.to(device).long()  # 必須是 long

            preds = model(imgs)              # (N, 4, 512, 512)
            loss  = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每 batch 第一張都可視化（方便看進度）
            if i == 0:
                pred_mask = torch.argmax(preds[0], dim=0)
                visualize_epoch(imgs[0].cpu(), masks[0].cpu(), pred_mask, f"epoch{epoch:02d}")

            print(f"Epoch {epoch:02d} [{i+1}/{len(loader)}] loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch:02d} 完成！平均 loss: {avg_loss:.4f}\n")

        # 儲存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(checkpoint_dir, "best_unet_model.pth")
            torch.save(model.state_dict(), path)
            print(f"新的最佳模型已儲存！loss = {avg_loss:.4f}")

        # 每 10 epochs 存一次備份
        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch}.pth"))

    print("訓練完成！最佳模型在：checkpoints/best_unet_model.pth")
    print("現在直接執行：streamlit run app.py 就能用了！")


if __name__ == "__main__":
    main()
