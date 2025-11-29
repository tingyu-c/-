# 發票記帳神器（Invoice Manager）

自動化解析台灣電子發票：UNet Segmentation、OCR、QR 解析、GPT 修補、Supabase 儲存、Streamlit 儀表板。

---

## 1. 專案介紹

本專案是一套能自動解析台灣電子發票的 AI 系統，可將「發票照片」轉換成可結構化資料，並存入資料庫，最後以儀表板呈現每月支出統計。

包含以下能力：

* 自動抽取三大欄位：發票號碼、日期、總金額
* OCR 搭配 GPT-fallback 修補錯誤
* 全圖 QR Code 掃描
* TEXT QR 品項解析（品名、數量、金額）
* 金額等比例調整
* 儲存到 Supabase（PostgreSQL）
* Streamlit 深色金融儀表板

---

## 2. 功能總覽

### 發票欄位擷取

* Tesseract OCR
* GPT-4o-mini 自動修補／補全／格式化
* 日期自動轉換（民國 ↔ 西元）

### TEXT QR 品項解析

* 支援多段式 TEXT QR（餐飲最常見）
* 自動過濾載具與贈品資訊
* 自動整理品名、數量、單價
* 合併同項目
* 將品項金額按比例調整，使其總和＝總金額

### 資料庫儲存（Supabase）

主表（發票）與子表（品項）使用正規化設計：

* invoices_data：儲存發票欄位
* invoice_items：儲存所有品項（有 FK）

### 儀表板

* 每月支出折線圖
* 類別支出圓餅圖
* 當月 KPI（總支出、成長率、最大支出類別）
* 依月份查看所有發票
* 點選發票 → 顯示品項明細
* 可刪除一張發票（含所有品項）
* 深色 UI（Finance Dashboard 風格）

---

## 3. 專案架構

```
📸 發票圖片
      │
      ▼
OCR + GPT-fallback
      │
      ▼
QR Code 全圖掃描
      │
      ▼
TEXT QR 品項解析器
      │
      ▼
資料清洗與金額等比例調整
      │
      ▼
Supabase（PostgreSQL）
      │
      ▼
Streamlit 深色儀表板
```

若啟用 UNet：

```
圖片 → UNet Segmentation → 分割位置 → OCR/GPT
```

---

## 4. 資料標註與模型訓練（UNet）

### JSON → Mask 轉換工具

使用 Labelme 標註後，以彩色 mask 匯出。

來源程式：
`json_to_mask.py` 

### Dataset（PyTorch）

將彩色 RGB mask 轉成三類（號碼、日期、金額）。

來源程式：
`dataset.py` 

### 模型（UNet）

使用純 PyTorch 實作的 Encoder–Decoder UNet。

來源程式：
`unet_model.py` 

### 訓練流程

* Resize 512x512
* CrossEntropyLoss
* AdamW
* 自動保存最佳模型
* 產生可視化圖像（原圖 / 標註 / 預測）

來源程式：
`train.py` 

---

## 5. 安裝方式

```bash
pip install -r requirements.txt
```

---

## 6. 執行方式

```bash
streamlit run app.py
```

---

## 7. 資料庫 Schema（Supabase）

### invoices_data（主表）

| 欄位           | 類型        |
| ------------ | --------- |
| id (PK)      | int8      |
| invoice_no   | text      |
| date         | text      |
| total_amount | float8    |
| category     | text      |
| note         | text      |
| created_at   | timestamp |

### invoice_items（子表）

| 欄位              | 類型     |
| --------------- | ------ |
| item_id (PK)    | int8   |
| invoice_id (FK) | int8   |
| name            | text   |
| qty             | float8 |
| price           | float8 |
| amount          | float8 |

---

## 8. 專案結構

```
invoice_project/
│
├── app.py                 # Streamlit 主程式
├── inference.py           # UNet 推論
├── train.py               # 模型訓練
├── dataset.py             # Dataset
├── unet_model.py          # UNet 架構
├── json_to_mask.py        # JSON → Mask 工具
│
├── images/                # 訓練影像
├── masks/                 # 彩色 mask
├── checkpoints/           # 模型權重
└── visualize/             # 訓練可視化
```

---

## 9. 未來可擴充功能

* 手機 App（Flutter / React Native）
* 自動生成 CSV / Excel 報表
* 每月預算與通知功能
* 支出預測（Time-series model）
* 店家分類模型（NLP + Clustering）

---

歡迎提出 Issue / PR。

---
