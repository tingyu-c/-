import torch
import numpy as np
from PIL import Image
from unet_model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

FIELDS = ["invoice_no", "date", "total_amount"]


def load_model(path):
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def preprocess(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = arr.transpose(2,0,1)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def run_unet(pil_img, checkpoint):
    model = load_model(checkpoint)

    x = preprocess(pil_img)

    with torch.no_grad():
        logits = model(x)[0]         # (3,512,512)
        prob = torch.sigmoid(logits)

    prob_np = prob.cpu().numpy()

    masks = {
        "invoice_no":   prob_np[0] > 0.25,
        "date":         prob_np[1] > 0.40,
        "total_amount": prob_np[2] > 0.30,
    }

    # 回推 bbox
    crops = {}
    for field, mask in masks.items():
        ys, xs = np.where(mask)
        if len(xs)==0: continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        ow, oh = pil_img.size
        x1 = int(x1 * ow / IMG_SIZE)
        x2 = int(x2 * ow / IMG_SIZE)
        y1 = int(y1 * oh / IMG_SIZE)
        y2 = int(y2 * oh / IMG_SIZE)

        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1-pad_x)
        y1 = max(0, y1-pad_y)
        x2 = min(ow, x2+pad_x)
        y2 = min(oh, y2+pad_y)

        crops[field] = pil_img.crop((x1,y1,x2,y2))

    return masks, crops
