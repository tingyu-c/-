# 🧾 Taiwan Invoice OCR System  
### **UNet + OCR + LLM for Automatic Invoice Field Extraction**

本專案實作一套 **台灣發票欄位自動擷取系統**，從原始影像到結構化 JSON 完全自動化。  
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
## 📂 Dataset Availability

本專案使用 **160 張台灣電子發票** 作為訓練資料，並由團隊自行使用  
**Labelme** 進行欄位標註（invoice_no / date / total_amount）。

由於資料中包含：

- 真實店家名稱  
- 發票號碼  
- 購買日期與金額  
- 可能涉及隱私或商業資訊  

因此 **無法於 GitHub 公開整套完整資料集**。

這也是許多包含實體文件、醫療資料、收據、票據的專案常見的限制。

### 🔒 Why the dataset cannot be published?

- 涉及真實消費資訊  
- 屬於企業或個人票據資料  
- 台灣發票屬於具隱私性文件，不適合公開大量原始影像  
- 可能造成資料外洩與法規風險  

基於以上原因，我們選擇不將完整 dataset 放上 GitHub。


## 🔧 How to Train the Model?

若您需要訓練自己的模型：

1. 準備自己的台灣電子發票資料  
2. 使用 Labelme 標註三個欄位：
   - `invoice_no`
   - `date`
   - `total_amount`
3. 使用本專案提供的工具：
   - `json_to_mask.py`：將 Labelme JSON 轉換為 segmentation masks  
   - `train.py`：進行 UNet 模型訓練  
---

## 📬 Need the dataset?

如需完整資料集進行研究用途，可透過 Issue 或 Email 與作者聯繫。  
我們可提供資料格式範本、標註流程指南，但 **無法提供完整未遮蔽資料影像**。
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

