# 📄 發票記帳神器 — Invoice Manager (UNet + OCR + QR + GPT + Supabase)

> 自動讀取台灣電子發票｜UNet 區塊定位｜Tesseract OCR｜GPT 修補｜全圖 QR 抓取品項｜Supabase 雲端記帳儀表板
>
> **支援：發票號碼、日期、總金額、自動品項解析、每月花費儀表板**

---

## 🚀 功能特色

### 🧠 1. UNet 發票欄位定位（深度學習）

模型可自動從完整發票圖片中定位：

* `invoice_no`（發票號碼）
* `date`（日期）
* `total_amount`（總金額）

使用 **UNet 512×512 segmentation**，你可在 `/checkpoints/best_unet_model.pth` 載入最佳模型。
模型結構：純 PyTorch、無 SPM 依賴
（檔案：[`unet_model.py`](./unet_model.py)）

---

### 🔍 2. OCR + GPT fallback

* 主要 OCR：Tesseract (`chi_tra+eng`)
* 辨識失敗 → 自動轉用 GPT-4o-mini 圖像辨識補齊（只回 JSON）

對於模糊、旋轉、印刷不清的發票非常有用。
（檔案：[`app_v41.py` / extract_invoice_meta](./app_v41.py)）

---

### 📦 3. 全圖 QR 掃描（品項自動解析）

同時支援：

* pyzxing（主要）
* OpenCV QRCodeDetector（備援）

自動解析餐飲業常見 **TEXT QR**，取得：

* 品名
* 數量
* 單價
* 自動等比例調整金額 → 使合計與發票總額一致

（功能檔案：`parse_text_qr_items()`、`detect_invoice_items()`）
（來源：[`app_v41.py`](./app_v41.py)）

---

### 🗄 4. Supabase 雲端記帳系統

自動寫入：

* `invoices_data`（發票主檔）
* `invoice_items`（品項子檔）

並提供：

* 每月花費折線圖
* 類別圓餅圖
* 當月 KPI（最高花費類別 / 成長率）
* 依月份檢索
* 點開單張發票查看細項
* 一鍵刪除（含所有品項）

---

### 🖥 5. 完整 Streamlit 介面

分成兩大分頁：

#### 📤 Tab 1 — 上傳與辨識

* 顯示原始影像
* UNet + OCR + GPT 結果
* TEXT QR 品項表格
* 類別 / 備註
* 儲存至資料庫

#### 📊 Tab 2 — 儀表板

* 每月花費、成長率、最大類別
* 月份切換
* 圖表視覺化
* 發票與品項清單

主要 UI 在：[`app_v41.py`](./app_v41.py)（中後段）

---

## 📁 專案結構

```
invoice_project/
│
├── app_v41.py              # Streamlit 主程式
├── inference.py            # UNet 推論流程（mask → bbox → crop）:contentReference[oaicite:4]{index=4}
├── unet_model.py           # PyTorch UNet 模型定義    :contentReference[oaicite:5]{index=5}
├── train.py                # 模型訓練程式（含 visualize）:contentReference[oaicite:6]{index=6}
├── dataset.py              # 資料集格式 + color mask → class mask:contentReference[oaicite:7]{index=7}
├── json_to_mask.py         # 將 Labelme JSON → 彩色 mask.png 生成器:contentReference[oaicite:8]{index=8}
│
├── images/                 # 原始訓練圖片
├── masks/                  # 彩色 segmentation masks
├── checkpoints/            # best_unet_model.pth 儲存位置
└── visualize/              # 訓練過程產生的可視化
```

---

## 🛠 安裝與環境設定

### 1. 安裝套件

```bash
pip install -r requirements.txt
```

必要套件包含：

* streamlit
* torch / torchvision
* pytesseract
* opencv-python
* supabase
* pyzxing
* plotly
* pillow
* numpy

### 2. Windows Tesseract 路徑

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 📘 資料集準備（UNet 訓練）

### Step 1 — 使用 LabelMe 標註 polygon

三種類別：

| Label        | Color (RGB) |
| ------------ | ----------- |
| invoice_no   | (255, 0, 0) |
| date         | (0, 255, 0) |
| total_amount | (0, 0, 255) |

### Step 2 — 轉成 mask

```bash
python json_to_mask.py
```

會在 `/masks/` 自動生成彩色 segmentation mask。

### Step 3 — 開始訓練

```bash
python train.py
```

訓練完成後：

```
checkpoints/best_unet_model.pth
```

---

## 🔮 UNet 推論（含 bbox + OCR crop）

範例：

```python
from PIL import Image
from inference import run_unet_inference

img = Image.open("invoice.jpg").convert("RGB")
mask, bboxes, crops = run_unet_inference(img, "checkpoints/best_unet_model.pth")
```

輸出：

* `mask`：512×512 類別矩陣
* `bboxes`：各欄位的 bounding boxes
* `crops`：切好的「發票號碼」、「日期」、「總金額」影像（可直接餵 OCR）

來源：[`inference.py`](./inference.py)

---

## 🧩 Streamlit 使用方式

### 啟動 APP

```bash
streamlit run app_v41.py
```

頁面包含：

* 發票圖片預覽
* UNet 分割結果
* OCR（Tesseract + GPT 修復）
* QR TEXT 品項解析
* 金額等比例校正
* 類別與備註輸入
* Supabase 上傳 / 刪除功能
* 每月儀表板與圓餅圖

---

## 🗄 Supabase Schema

### invoices_data

| 欄位           | 型態        |
| ------------ | --------- |
| id           | bigint PK |
| invoice_no   | text      |
| date         | date      |
| total_amount | float     |
| category     | text      |
| note         | text      |

### invoice_items

| 欄位         | 型態        |
| ---------- | --------- |
| id         | bigint PK |
| invoice_id | fk        |
| name       | text      |
| qty        | float     |
| price      | float     |
| amount     | float     |

---

## 🎯 Roadmap

* [ ] 手機版 UI
* [ ] 自動同步載具發票
* [ ] 自動行程消費分類（AI）
* [ ] OCR 景深模糊修正
* [ ] 導入更強 segmentation backbone

---

## 📜 License

This project is open-source under MIT License.


---

歡迎提出 Issue / PR。

---
