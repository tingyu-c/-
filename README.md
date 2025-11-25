📄 Taiwan E-Invoice OCR System
UNet + OCR + LLM for Automatic Invoice Field Extraction

本專案是一個完整的台灣電子發票自動化資訊擷取系統，結合 深度學習語意分割（UNet）、Tesseract OCR 與 Large Language Model（LLM），能從發票影像中準確擷取：

🧾 發票號碼（invoice_no）

📅 發票日期（date）

💰 總金額（total_amount）

本系統由團隊自行標註 160 張臺灣電子發票，並完成從訓練到系統部署的完整工作流程。

🚀 Features
✔ 1. UNet Segmentation

使用 Labelme 標註 3 類欄位區塊

UNet 模型負責定位「發票號碼／日期／金額」的影像位置

訓練過程中會自動輸出可視化結果（真值 Mask vs 預測 Mask）

✔ 2. OCR 文字辨識

使用 Tesseract OCR 逐區塊辨識

減少背景干擾、提高辨識準確度

✔ 3. LLM 後處理

LLM 負責：

修正 OCR 誤判（0/O、1/l 等）

日期格式轉換（例：112/01/03 → 2023-01-03）

金額格式化與合理性檢查

最終輸出標準化 JSON

✔ 4. Web Demo

使用 Streamlit 打造互動式介面：

上傳發票圖片

顯示 segmentation mask

顯示裁切後的欄位區塊

顯示 OCR + LLM 最終輸出
🏷️ Data Annotation (Labelme)

團隊使用 Labelme 標記以下欄位：

Class ID	Label
0	background
1	invoice_no
2	date
3	total_amount

轉換 JSON → mask：

python json_to_mask.py

🧠 Model Training (UNet)

訓練指令：

python train.py


訓練特點：

data augmentation

每個 epoch 產生可視化輸出：

epochX_img.png

epochX_true_mask.png

epochX_pred_mask.png

自動儲存：

unet_epochX.pth

best_unet_model.pth

🔍 Inference Flow (Segmentation → OCR → LLM)

推論流程整合於 inference.py：

UNet 產生 segmentation mask

自動裁切三大欄位區域

使用 Tesseract OCR 辨識裁切內容

使用 LLM 校正格式並生成結構化 JSON

🖥️ Streamlit Web Demo

啟動：

streamlit run app.py


Demo 功能：

上傳發票圖片

顯示 UNet segmentation 結果

顯示三大欄位裁切區域

顯示 OCR + LLM 的解析結果

輸出 JSON

📈 Training Visualization

訓練過程可於 visualize/ 查看 segmentation：

發票欄位定位是否成功

mask 是否收斂

模型是否正確學會三類欄位區域

🛠️ Technologies Used
技術	說明
UNet	三類欄位的 segmentation
Labelme	手動標註與 polygon 標記
PyTorch	模型訓練
Tesseract OCR	區塊文字辨識
OpenAI LLM	文字校正與 JSON 輸出
Streamlit	Web Demo
📦 Installation
1. 安裝必要套件
pip install -r requirements.txt


若無 requirements.txt，我能替你生成。

2. 安裝 Tesseract OCR（Windows）

請安裝官方版本並加入 PATH：

C:\Program Files\Tesseract-OCR\tesseract.exe

📜 Example Output (JSON)
{
  "invoice_no": "AB12345678",
  "date": "2023-01-05",
  "total_amount": 268
}

🤝 Contribution

歡迎 issue / PR！
如果你想新增：

YOLO-based 文字偵測

LLM-based OCR end-to-end

多欄位擴增（店名、品項）

FastAPI 版本

都可以提出。

📄 License

MIT License
