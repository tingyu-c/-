# 📘 發票記帳神器 v42

**UNet Segmentation + OCR + GPT-4o-mini Fallback + QR 掃描 + Supabase 儲存**

本專案提供一個 **全自動化的台灣電子發票辨識系統**：
從一張圖片開始 → 自動找出欄位位置 → OCR → 修正常見錯誤 → 解析 TEXT QR → 儲存到雲端資料庫 → 並提供完整儀表板分析。

支援：

* 🟥 **發票號碼 segmentation（UNet）**
* 🟩 **日期 segmentation（UNet）**
* 🟦 **總金額 segmentation（UNet）**
* 🔍 **Tesseract OCR**
* 🤖 **GPT-4o-mini fallback（OCR 錯誤自動修正）**
* 🧾 **TEXT QR 全圖掃描與品項解析**
* 🗄 **Supabase 資料庫儲存**
* 📊 **Streamlit 儀表板**
* 📤 **一鍵匯出與統計報表**

---

## 📁 專案結構

```
project/
│
├── app_v42.py                          # Streamlit 主程式（上傳/辨識/儀表板）
│
├── train.py                             # UNet 訓練（Dice+Focal + bias init）
├── unet_model.py                        # UNet 架構（3-channel multi-label）
├── dataset.py                           # Dataset（讀 fixed_images + fixed_masks）
├── inference.py                         # segmentation 推論 + bbox 回推
│
├── rescue_masks_from_json_final.py      # 自動修復 LabelMe JSON mask → (H,W,3)
│
├── fixed_images/                        # 統一尺寸圖片
├── fixed_masks/                         # (H,W,3) 多標籤 segmentation mask (.npy)
├── checkpoints/
│   └── best_unet.pth                   # 訓練最佳模型
│
└── visualize/                           # 每個 epoch 的 segmentation 可視化
```

---

## 🧠 模型設計：UNet + Multi-Label Segmentation

本專案採用 **3-channel multi-label segmentation**，
每個通道代表一項欄位（可同時重疊，避免 cross-entropy 的壓制問題）。

| Channel | 欄位   | 顏色       |
| ------- | ---- | -------- |
| 0       | 發票號碼 | 🟥 red   |
| 1       | 日期   | 🟩 green |
| 2       | 總金額  | 🟦 blue  |

Loss 採用：

* **DiceLoss 0.85**
* **FocalLoss 0.15（正樣本權重 α=0.8）**

適用於：

* 小文字 segmentation
* 佔畫面僅 0.1%–1% 的極小區域
* 背景佔比極大（>99% 的 pixel）

---

## 🛠 訓練：train.py

```
python train.py
```

訓練流程：

* 自動載入 `fixed_images/` 與 `fixed_masks/`
* 多標籤 segmentation 訓練
* 每個 epoch 可視化 true/pred mask
* 自動保存最佳模型 `checkpoints/best_unet.pth`

---

## 🧼 標註與 Mask 修復（必做）

使用 LabelMe 標註後，請執行以下腳本將 JSON → (H,W,3) segmentation mask：

```
python rescue_masks_from_json_final.py
```

輸出：

* `fixed_images/xxx.jpg`
* `fixed_masks/xxx.npy`（三通道 mask）

這是 UNet 訓練的唯一正確格式。

---

## 🔍 推論：inference.py

使用 segmentation 推論 + OCR 切圖：

```python
from inference import run_unet
masks, crops = run_unet(pil_image, "checkpoints/best_unet.pth")
```

各項裁切影像（可送入 OCR）：

```python
crops["invoice_no"]
crops["date"]
crops["total_amount"]
```

---

## 🤖 OCR + GPT Fallback

app 會使用：

1. **UNet 找出欄位位置**
2. **Tesseract OCR 辨識**
3. 若未成功 → 自動啟動 **GPT-4o-mini（圖片 + 欄位）** 補齊欄位

GPT 回傳格式：

```json
{
  "invoice_no": "AB12345678",
  "date": "2025-01-10",
  "total_amount": "520"
}
```

---

## 🧾 TEXT QR 掃描與品項解析

使用：

* `pyzxing`（主力）
* `OpenCV detectAndDecodeMulti`（備援）

可支援：

* 超髒亂 TEXT QR
* 載具碼 / 品項 / 加購 / 贈品 / 小計
* 自動清洗：過濾噪音、合併相同品名、數量/單價推算
* 最終金額以總額 → **等比例調整**

輸出：

| name | qty | price | amount |
| ---- | --- | ----- | ------ |

---

## 💾 Supabase 儲存

儲存兩個表：

### `invoices_data`

| 欄位           | 說明   |
| ------------ | ---- |
| invoice_no   | 發票號碼 |
| date         | 日期   |
| total_amount | 總金額  |
| category     | 類別   |
| note         | 備註   |

### `invoice_items`

| 欄位         | 說明   |
| ---------- | ---- |
| invoice_id | 主檔連結 |
| name       | 品名   |
| qty        | 數量   |
| price      | 單價   |
| amount     | 小計   |

---

## 📊 儀表板功能（Tab 2）

包含：

* 當月支出
* 月成長率
* 本月最大類別
* 各類別圓餅圖
* 月支出折線圖
* 特定月份檢視
* 發票明細列表
* 一鍵刪除發票（含子項目）

---

## 🚀 部署

### 1. 安裝依賴：

```
pip install -r requirements.txt
```

### 2. 設定 Streamlit secrets (`.streamlit/secrets.toml`)

```
SUPABASE_URL="https://xxxxx.supabase.co"
SUPABASE_KEY="your_anon_key"
```

### 3. 啟動 App

```
streamlit run app_v42.py
```

---

## 🗺 Roadmap

* [ ] 加入手機拍照自動裁切（Perspective transform）
* [ ] 加入 RNN/Transformer OCR 替代 Tesseract
* [ ] 發票載具自動對帳
* [ ] 商家分類自動化（AI 分類器）
* [ ] 內建多發票批次上傳

---

## 🪪 License

MIT License

--

