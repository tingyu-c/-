# ============================================================
# inference.py — v11 (Pure PyTorch UNet, no SMP)
# Segmentation → BBox → Crops for OCR
# ============================================================

import torch
import numpy as np
from PIL import Image
from unet_model import UNet


# -----------------------------
# 設定
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512     # 與 train 保持一致
NUM_CLASSES = 4    # 0-bg, 1-no, 2-date, 3-total_amount

CLASS_ID_TO_FIELD = {
    1: "invoice_no",
    2: "date",
    3: "total_amount",
}


# -----------------------------
# 載入模型
# -----------------------------
def load_unet_model(checkpoint_path):
    model = UNet(n_channels=3, n_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# 影像前處理（512）
# -----------------------------
def preprocess(pil_img):
    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(pil_img) / 255.0
    arr = arr.transpose(2, 0, 1)  # (C, H, W)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)


# -----------------------------
# mask argmax
# -----------------------------
def postprocess_mask(mask_logits):
    """
    mask_logits shape: (4, H, W)
    return: mask (H, W) with class id
    """
    return np.argmax(mask_logits, axis=0)


# -----------------------------
# mask → bbox
# -----------------------------
def mask_to_bboxes(mask):
    """
    回傳 { "invoice_no": (x1,y1,x2,y2), ... }
    """
    bboxes = {}

    for cid, field in CLASS_ID_TO_FIELD.items():
        ys, xs = np.where(mask == cid)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bboxes[field] = (x1, y1, x2, y2)

    return bboxes


# -----------------------------
# bbox crop （從512轉回原圖大小）
# -----------------------------
def crop_bbox(original_img, bbox):
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    orig_w, orig_h = original_img.size

    # 正確縮放（這才是對的）
    x1 = int(x1 * orig_w / IMG_SIZE)
    x2 = int(x2 * orig_w / IMG_SIZE)
    y1 = int(y1 * orig_h / IMG_SIZE)
    y2 = int(y2 * orig_h / IMG_SIZE)

    # 加上 10~15% padding + 邊界保護
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(orig_w, x2 + pad_x)
    y2 = min(orig_h, y2 + pad_y)

    # 最小尺寸保護
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    return original_img.crop((x1, y1, x2, y2))

# -----------------------------
# 主流程：Segmentation + BBox + Crops
# -----------------------------
def run_unet_inference(pil_img, checkpoint_path):
    """
    回傳：
    mask, bboxes, crops(dict)
    """
    # 1. 模型
    model = load_unet_model(checkpoint_path)

    # 2. 預處理
    x = preprocess(pil_img)

    # 3. 推論
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()  # shape (4,512,512)

    # 4. mask (argmax)
    mask = postprocess_mask(pred)

    # 5. bbox
    bboxes = mask_to_bboxes(mask)

    # 6. crop
    crops = {}
    for field, box in bboxes.items():
        crops[field] = crop_bbox(pil_img, box)

    return mask, bboxes, crops


# -----------------------------
# Optional: mask 疊圖（debug）
# -----------------------------
def visualize_mask(pil_img, mask):
    color_map = {
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    arr = np.array(pil_img).copy()

    for cid, color in color_map.items():
        ys, xs = np.where(mask == cid)
        arr[ys, xs] = color

    return Image.fromarray(arr)
