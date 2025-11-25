

import os
import json
import numpy as np
from PIL import Image, ImageDraw

input_dir = r"C:\Users\user\Desktop\invoice_project\labelme_json"      # labelme json 資料夾
output_dir = r"C:\Users\user\Desktop\invoice_project\data\masks"      # 輸出 mask

def json_to_mask(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 雷：有些 points 是字串，要強制轉 float
    def fix_points(points):
        return [(float(p[0]), float(p[1])) for p in points]

    img_w = int(data["imageWidth"])
    img_h = int(data["imageHeight"])
    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in data.get("shapes", []):
        pts = fix_points(shape["points"])
        draw.polygon(pts, fill=1, outline=1)

    return np.array(mask, dtype=np.uint8)

def main():

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(input_dir, filename)

        try:
            mask = json_to_mask(json_path)

            base = filename.rsplit(".", 1)[0]
            mask_path = os.path.join(output_dir, base + ".png")

            Image.fromarray(mask * 255).save(mask_path)
            print("✔ 生成：", mask_path)

        except Exception as e:
            print("❌ 失敗：", filename, "| 錯誤：", e)

    print("\n🎉 全部 JSON → mask 轉換完成！")

if __name__ == "__main__":
    main()
