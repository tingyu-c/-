# 🧾 Taiwan E-Invoice OCR System  
### **UNet + OCR + LLM for Automatic Invoice Field Extraction**

本專案實作一套 **台灣電子發票欄位自動擷取系統**，從原始影像到結構化 JSON 完全自動化。  
系統結合 **UNet 影像分割、Tesseract OCR、與 LLM 後處理**，可精準擷取：

- **發票號碼（invoice_no）**  
- **發票日期（date）**  
- **總金額（total_amount）**

我們團隊使用 Labelme 手動標註 **160 張台灣電子發票**，完成系統訓練與部署。

---

## 📌 **Project Overview**

```
Taiwan E-Invoice OCR System
│
├── 1️⃣ UNet Segmentation
│      ├── invoice_no 區域
│      ├── date 區域
│      └── total_amount 區域
│
├── 2️⃣ OCR (Tesseract)
│      └── 對各區域進行文字辨識
│
└── 3️⃣ LLM Post-processing
       ├── 修正 OCR 錯字
       ├── 格式化日期
       ├── 金額合理性檢查
       └── 輸出結構化 JSON
```

（使用「純文字架構圖」確保在 GitHub 不會跑版）

---

## 🚀 **Features**

### ✔ UNet Segmentation  
- 自行標註 160 張 Labelme polygon  
- 模型輸出三種欄位的 segmentation mask  
- 訓練期間自動可視化結果（true mask / pred mask）

### ✔ OCR Recognition  
- 使用 **Tesseract OCR**  
- 針對每個 Segmentation 區域裁切後辨識  
- 大幅提升 OCR 精準度

### ✔ LLM Post-processing  
LLM 用於：

- 修正 OCR 誤判（1/7、0/O 等）  
- 日期補格式（例：`112/01/03` → `2023-01-03`）  
- 金額數字清洗  
- 輸出標準化 JSON

### ✔ Streamlit Web Demo  
使用者可以：

- 上傳發票  
- 檢視 segmentation mask  
- 檢視欄位裁切 crop  
- 查看 OCR + LLM 的最終解析結果  

---

## 📁 **Project Structure**

```
invoice_project/
│
├── data/
│   ├── images/          # 原始發票圖片
│   ├── masks/           # 由 labelme JSON 轉換的 segmentation mask
│
├── dataset.py           # PyTorch Dataset + augmentation
├── json_to_mask.py      # JSON → mask 轉換工具
├── train.py             # UNet 訓練（含可視化）
├── unet_model.py        # UNet 模型結構
├── inference.py         # Segmentation + OCR + LLM 推論
├── app.py               # Streamlit Web App
└── checkpoints/         # 模型權重
```

---

## 🛠️ **Installation**

### 1. Clone the repo
```bash
git clone https://github.com/<yourname>/invoice-ocr-system
cd invoice-ocr-system
```

### 2. Install dependencies
（如需，我可以幫你產生 requirements.txt）

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR（Windows）
請安裝後確認路徑如下：

```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## 🎯 **Training UNet**

```bash
python train.py
```

訓練時會自動輸出：

```
visualize/
  ├── epoch1_img.png
  ├── epoch1_true_mask.png
  └── epoch1_pred_mask.png
```

以及：

```
checkpoints/
  ├── unet_epoch1.pth
  ├── unet_epoch2.pth
  └── best_unet_model.pth
```

---

## 🔍 **Run Inference**

```bash
python inference.py
```

輸出會包含：

- segmentation mask  
- bbox  
- crop images  
- OCR raw text  
- LLM 修正後 JSON  

---

## 🖥️ **Streamlit Web Demo**

```bash
streamlit run app.py
```

Demo 包含：

- 上傳圖片  
- 自動 segmentation  
- OCR + LLM 結果  
- 結構化資料顯示  

---

## 📦 **Example Output**

```json
{
  "invoice_no": "AB12345678",
  "date": "2023-01-05",
  "total_amount": 268
}
```

---

## 🧩 **Tech Stack**

| Component | Technology |
|----------|------------|
| Segmentation | UNet (PyTorch) |
| Annotation | Labelme |
| OCR | Tesseract |
| LLM | OpenAI / gpt-4.1-mini / GPT-5 |
| Web UI | Streamlit |
| Data Augmentation | Albumentations |

---

## 🤝 **Contributions**

歡迎提出 issue / PR！  
如果你想新增：

- YOLOv8 / YOLO-World for text detection  
- OCR-free end-to-end 模型  
- FastAPI REST API  
- Cloud 部署（Railway / Render）  

都非常歡迎。

---

## 📄 License

MIT License.

---

## ⭐ Support

如果這個專案對你有幫助，  
請幫忙點個 ⭐️ 支持！

