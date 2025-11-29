🧾 發票記帳神器（Invoice Manager）
UNet Segmentation + OCR + 全圖 QR 解碼 + GPT 修補 + Streamlit + Supabase
Automatically extract Taiwanese electronic invoice fields & item details
📌 專案簡介

這是一套 全自動化「台灣電子發票解析＋記帳系統」。
只要上傳發票照片，就能自動：

辨識發票三大欄位

發票號碼

日期（自動轉民國 → 西元）

總金額

擷取 TEXT QR 內的品項、數量、單價

旗標載具、贈品、點數，並自動過濾噪音字串

將所有品項依比例調整金額（總額一致）

存入 Supabase（主檔 + 子檔）

以深色金融儀表板呈現每月花費與類別統計

支援電腦版即時操作，也適合作為 Side Project 展示。

🚀 Demo 功能展示
✔ 上傳發票 → 自動辨識

支援任意拍攝角度

自動辨識三大欄位

TEXT QR 品項解析

✔ 儀表板

每月支出折線圖

類別支出圓餅圖

當月 KPI（成長率 / 最大支出類別）

查看某月所有發票與品項

一鍵刪除發票（含所有子項）

🧠 系統架構總覽
📸 圖片上傳
       ↓
UNet Segmentation（可選） — (PyTorch)
       ↓
OCR (Tesseract) + GPT-4o-mini fallback
       ↓
全圖 QR 偵測（pyzxing + OpenCV detectMulti）
       ↓
TEXT QR 品項解析器（多階段清洗 + 錯誤容忍）
       ↓
金額等比例調整（避免總額不一致）
       ↓
Supabase（PostgreSQL）
       ↓
Streamlit UI（深色專業風儀表板）

🔍 核心技術亮點
1️⃣ UNet Segmentation（訓練代碼）

為了解析發票位置，本專案提供完整 UNet 分割模型訓練流程：

JSON → 彩色 mask（Labelme）

PyTorch Dataset

UNet (Encoder-Decoder)

50 epochs 訓練

模型儲存與可視化

最後可切換是否使用 UNet (預設Unet)

📄 程式碼：
dataset.py 

dataset


train.py 

train


unet_model.py 

unet_model


inference.py 

inference

2️⃣ OCR → GPT Fallback 校正

使用 Tesseract OCR 抓取資訊，再由 GPT-4o-mini:

自動補全錯誤辨識

解析日期格式

自動抽取總金額

回傳純 JSON

保證三欄位一定存在

3️⃣ 全圖 QR 偵測（雙引擎）

pyzxing：精準強

OpenCV detectAndDecodeMulti：輔助

自動合併結果

支援包含：

TEXT QR（主要）

載具 QR

點數 QR

亂碼 QR

贈品 / 折抵 QR

4️⃣ 強化 TEXT QR 品項解析器

品項內容常見：

**:泡菜豚中:1:155:海帶湯:1:35:...


本專案處理：

移除載具噪音、贈品欄位

清洗特殊字元（※、＊、＠、＄）

正規化品名

合併同項目

按金額排序

金額自動調整到總額一致

5️⃣ Supabase（PostgreSQL）資料庫設計

主表：

欄位	類型
id (PK)	int8
invoice_no	text
date	text
total_amount	float8
category	text
note	text

子表（invoice_items）：

欄位	類型
item_id (PK, 自動遞增)	int8
invoice_id (FK)	int8
name	text
qty	float8
price	float8
amount	float8
6️⃣ Streamlit 深色儀表板 UI

（含月支出折線圖、圓餅圖、KPI、發票細項、刪除功能）

主要程式碼：
app_v41.py / app_v42.py 

app_v41

📦 安裝與執行
安裝依賴：
pip install -r requirements.txt

執行：
streamlit run app.py

🔧 專案結構
invoice_project/
│
├── app.py                 # Streamlit 主程式
├── inference.py           # UNet 推論流程
├── train.py               # UNet 訓練
├── dataset.py             # 資料集建立
├── unet_model.py          # U-Net 架構
├── json_to_mask.py        # 標註 JSON → 彩色 Mask
│
├── images/                # 原始影像
├── masks/                 # 彩色 mask
├── checkpoints/           # 模型儲存
└── visualize/             # 訓練可視化

🧪 未來可加入功能

手機版前端（React Native / Flutter）

匯出 Excel / CSV

每月預算、自動提醒

消費趨勢預測模型（Time-Series）

自動判斷店家屬性（NLP / Clustering）


歡迎提出 issue / PR，一起讓台灣發票 AI 更好用！
