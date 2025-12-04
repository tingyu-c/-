# ============================================================
# app.py — 發票記帳神器（UNet + OCR + 全圖QR + GPT Fallback + Supabase）
# ============================================================
import os
import io
import re
import json
import base64
import numpy as np
from uuid import uuid4
from PIL import Image
import streamlit as st
import pandas as pd
import cv2
from supabase import create_client
import openai
import plotly.express as px
from typing import Dict
from PIL import Image
import numpy as np
from openai import OpenAI
from collections import Counter
import time
import pandas as pd
import tempfile
from datetime import datetime

# ========= 全域 EasyOCR Reader（只初始化一次，速度提升 10 倍） =========
import easyocr
from pyzxing import BarCodeReader
# 全域初始化（整個程式只跑一次，超快）
zxing_reader = BarCodeReader()


if "GLOBAL_EASYOCR_READER" not in st.session_state:
    st.session_state.GLOBAL_EASYOCR_READER = easyocr.Reader(
        ['en'], gpu=False  # 你沒有 GPU → 一定要設定 gpu=False
    )

reader = st.session_state.GLOBAL_EASYOCR_READER

from pyzxing import BarCodeReader

zxing_reader = BarCodeReader()


def parse_left_qr(left_qr_text):
    if not left_qr_text or ":" not in left_qr_text:
        return {}

    try:
        body = left_qr_text.split(":")[0]

        if len(body) < 37:
            return {}

        # 正確電子發票格式 offset（財政部規範）
        inv_no = body[0:10]
        roc_date = body[10:17]
        random_code = body[17:21]

        # 核心修正（Tammy 你現在最需要的）
        sales_hex = body[21:29]      # 未稅金額 HEX
        total_hex = body[29:37]      # 含稅金額 HEX ← 你抓錯位置在這！

        # 日期：民國 → 西元
        year = 1911 + int(roc_date[0:3])
        month = int(roc_date[3:5])
        day = int(roc_date[5:7])
        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        total_amount = int(total_hex, 16)

        return {
            "invoice_no": inv_no,
            "date": date_str,
            "random_code": random_code,
            "total_amount": str(total_amount)
        }

    except Exception as e:
        st.warning(f"左 QR 解析錯誤：{e}")
        return {}


def parse_text_qr(text_qr):
    """
    解析右側 TEXT QR：
    格式：
        **:品名:數量:單價:品名:數量:單價...
    """

    if not text_qr or not text_qr.startswith("**"):
        return []

    # 乾淨化：去掉開頭 **
    clean = text_qr.lstrip("*")
    parts = clean.split(":")

    # 去掉第一段空品名
    parts = parts[1:] if parts and parts[0] == "" else parts

    items = []
    buf = []
    for p in parts:
        if re.match(r"^\d+(\.\d+)?$", p):
            buf.append(p)
        else:
            # 遇到品名時重新起一段
            buf.append(p)

        # 每 3 個一組：品名、數量、價格
        if len(buf) == 3:
            name = buf[0]
            qty = int(float(buf[1]))
            price = int(float(buf[2]))
            items.append({
                "name": name,
                "qty": qty,
                "price": price,
                "subtotal": qty * price
            })
            buf = []

    return items

def zxing_scan_raw(uploaded_file):
    raw_bytes = uploaded_file.getvalue()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fp:
        fp.write(raw_bytes)
        temp_path = fp.name

    result = reader.decode(temp_path)
    return result


# 🔧 全圖 QR 辨識
from pyzbar.pyzbar import decode

def extract_from_qr_zxing(pil_img: Image.Image):
    """
    只做單張圖 pyzxing 解碼（不做多重增強）
    回傳：list of raw_text（可能是多個 QR）
    """

    # 1. 先把 PIL 轉成暫存檔（pyzxing 必須吃檔案路徑）
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fp:
        temp_path = fp.name
        pil_img.save(temp_path)

    # 2. pyzxing decode
    try:
        results = zxing_reader.decode(temp_path)
    except Exception as e:
        return []

    if not results:
        return []

    # results 是 list of dict：{"raw": b"...", "text": "..."}
    decoded_texts = []
    for r in results:
        raw = r.get("raw")
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8", errors="ignore")
            except:
                raw = ""
        decoded_texts.append(raw)

    return decoded_texts

def clean_invoice_no(raw: str) -> str:
    """
    清理各種發票號碼格式：
    支援：
    - TL-42103447
    - TL42103447
    - TL 42103447
    - TL：42103447
    - TL－42103447（全形 dash）
    - OCR 抓到的英數混雜符號
    - 混入其他亂碼的情況

    最終輸出：AA99999999（2 英文字 + 8 數字）
    """
    if not raw or not isinstance(raw, str):
        return ""

    # 統一格式
    raw = raw.upper().strip()
    
    # 移除所有非字母數字（包含 dash、空白、全形符號）
    raw = re.sub(r"[^A-Z0-9]", "", raw)

    # 直接找標準格式（最重點）
    match = re.search(r"[A-Z]{2}\d{8}", raw)
    if match:
        return match.group(0)

    # fallback：拆字母 + 數字重新組合
    letters = re.findall(r"[A-Z]", raw)
    digits = re.findall(r"\d", raw)

    if len(letters) >= 2 and len(digits) >= 8:
        return "".join(letters[:2]) + "".join(digits[:8])

    # 能救多少算多少：至少保持乾淨，不報錯
    return raw



def clean_date(text: str) -> str:
    """
    嘗試把 OCR 讀出的日期格式化成 YYYY-MM-DD
    支援：
    - 2025/01/10
    - 2025-1-5
    - 2025.01.05
    - 1140105（民國）
    """
    if not text:
        return ""

    text = text.strip()

    # ---------- 民國格式（如 1140105）----------
    if re.fullmatch(r"\d{7}", text):
        try:
            roc = int(text[:3]) + 1911
            m = int(text[3:5])
            d = int(text[5:7])
            return f"{roc:04d}-{m:02d}-{d:02d}"
        except:
            pass

    # ---------- 西元常見分隔符 ----------
    text = text.replace(".", "-").replace("/", "-")
    parts = text.split("-")

    if len(parts) == 3:
        try:
            y = int(parts[0])
            m = int(parts[1])
            d = int(parts[2])
            return f"{y:04d}-{m:02d}-{d:02d}"
        except:
            pass

    # 其他無法解析
    return ""


def parse_qr_invoice(pil_img: Image.Image):
    """
    用 pyzbar 找左右 → 用 pyzxing 解內容
    """
    import numpy as np, cv2

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 1. pyzbar 取位置
    qrs = decode(img)
    if not qrs:
        return "", ""

    # 取出 (x, raw_text)
    qr_boxes = []
    for q in qrs:
        x = q.rect.left
        txt = q.data.decode("utf-8", errors="ignore").strip()
        qr_boxes.append((x, txt))

    qr_boxes.sort(key=lambda z: z[0])  # ← 左右排序

    # 2. pyzxing 取內容（多重影像增強）
    zx_texts = extract_from_qr_zxing(pil_img)

    # 配對 raw_text → 特徵修正
    def best_match(raw):
        # TEXT QR（右）判斷：包含品項格式
        if raw.startswith("**") or ":" in raw:
            return raw

        # raw 太短或破損 → 用 zx 內容補
        for zx in zx_texts:
            if zx and zx != raw:
                return zx

        return raw

    if len(qr_boxes) == 1:
        return best_match(qr_boxes[0][1]), ""

    left_qr  = best_match(qr_boxes[0][1])
    right_qr = best_match(qr_boxes[1][1])

    return left_qr, right_qr

def clean_invoice_no(raw: str) -> str:
    """清洗 OCR or GPT 的發票號碼，只保留 2 碼英文 + 8 碼數字"""
    if not raw:
        return ""

    # 統一格式：去掉空白、奇怪符號
    raw = raw.strip().upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)

    # 如果太短 → 直接回傳
    if len(raw) < 10:
        return raw

    # 找 2 英文 + 8 數字 的 pattern
    match = re.search(r"[A-Z]{2}\d{8}", raw)
    if match:
        return match.group(0)

    # 找不到 → 最後嘗試強制切割前 10 碼
    return raw[:10]


# ------------------------------
# Layout
# ------------------------------
st.set_page_config(page_title="發票記帳神器", layout="wide")
# === 背景儲存狀態初始化 ===
if "save_status" not in st.session_state:
    st.session_state.save_status = "idle"      # idle / saving / success / error
if "last_save_time" not in st.session_state:
    st.session_state.last_save_time = None
if "last_error" not in st.session_state:
    st.session_state.last_error = ""

# ------------------------------
# Sidebar：API Key 設定
# ------------------------------
st.sidebar.header("🔑 OpenAI API Key 設定")
apikey = st.sidebar.text_input("請輸入 OpenAI API Key：", type="password", key="apikey_input")
if apikey:
    st.sidebar.success("API Key 已讀取 ✔")
else:
    st.sidebar.warning("尚未輸入 API Key")

# ------------------------------
# Import UNet inference
# ------------------------------
from inference import run_unet

# ============================================================
# Supabase 初始化
# ============================================================
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.sidebar.success("Supabase 連線成功 ✔")
    except Exception as e:
        st.sidebar.error(f"Supabase 連線失敗：{e}")
else:
    st.sidebar.warning("尚未設定 Supabase secrets")

def extract_invoice_meta(uploaded_file, pil_img, checkpoint_path, apikey):


    meta = {"invoice_no": "", "date": "", "total_amount": ""}

    # ============================================================
    # Step 0：ZXing 掃描 左 / 右 QR（唯一正確方法）
    # ============================================================
    qr_left, qr_right = parse_qr_invoice(pil_img)


    st.subheader("🔍 QR Debugger")
    st.write("📎 左 QR:", qr_left)
    st.write("📎 右 QR:", qr_right)

    # ============================================================
    # Step 1：UNet Segmentation
    # ============================================================
    try:
        from inference import run_unet
        masks, crops = run_unet(pil_img, checkpoint_path)
    except Exception as e:
        st.error(f"UNet 發生錯誤：{e}")
        crops = {}

    # ============================================================
    # Step 2：GPT ROI 讀金額（ROI 最準 → 但仍低於左 QR）
    # ============================================================
    amount_crop = crops.get("total_amount")
    gpt_roi_amount = ""

    if amount_crop is not None:
        gpt_roi_amount = gpt_read_amount_from_roi(apikey, amount_crop)

        if gpt_roi_amount.isdigit():
            meta["total_amount"] = gpt_roi_amount
            st.success(f"✔ 使用 GPT ROI 金額：{gpt_roi_amount}")
        else:
            st.warning("⚠ GPT ROI 金額失敗")

    # ============================================================
    # Step 3：解析左 QR（金額 100% 正確 → 永遠最高優先）
    # ============================================================
    info_left = parse_left_qr(qr_left)

    if info_left.get("total_amount"):
        meta["invoice_no"] = clean_invoice_no(info_left.get("invoice_no", meta["invoice_no"]))
        meta["date"] = info_left.get("date", meta["date"])

        # 左 QR 100% 最準 → 覆蓋 GPT ROI 金額
        meta["total_amount"] = str(info_left["total_amount"])

        st.success(f"✔ 使用 左 QR 金額（最高優先，最準確）：{meta['total_amount']}")
    else:
        st.warning("⚠ 左 QR 無法解析 → 使用下一順位")

    # ============================================================
    # Step 4：解析右 QR（TEXT QR 品項）
    # ============================================================
    items = parse_text_qr(qr_right)

    if items:
        sum_items = sum([it["subtotal"] for it in items])
        st.write(f"📦 TEXT QR 品項加總：{sum_items}")

        # 左 QR（或 GPT ROI）一致性檢查
        if meta["total_amount"] and str(sum_items) == meta["total_amount"]:
            st.info("✔ 右 QR 品項金額與左 QR 一致")
        else:
            st.warning("⚠ 右 QR 品項金額與左 QR 不一致")

        # 若前面完全沒有金額 → 才用右 QR 金額
        if not meta["total_amount"]:
            meta["total_amount"] = str(sum_items)
            st.success(f"✔ 使用右 QR 品項金額：{meta['total_amount']}")
    else:
        st.warning("⚠ TEXT QR 無品項或格式錯誤")

    # ============================================================
    # Step 5：OCR fallback（補 invoice_no / date）
    # ============================================================
        invoice_no_crop = crops.get("invoice_no")
        date_crop = crops.get("date")
        
        # ---------- 補發票號碼 ----------
        if not meta.get("invoice_no") and invoice_no_crop is not None:
            try:
                ocr_no = ocr_easy(invoice_no_crop)
                meta["invoice_no"] = clean_invoice_no(ocr_no)
            except Exception as e:
                st.warning(f"OCR 發票號碼失敗：{e}")
        
        # ---------- 補日期 ----------
        if not meta.get("date") and date_crop is not None:
            try:
                ocr_date = ocr_easy(date_crop)
                meta["date"] = clean_date(ocr_date)
            except Exception as e:
                st.warning(f"OCR 日期失敗：{e}")


    # ============================================================
    # Step 6：GPT 全圖 fallback（不能覆蓋金額！）
    # ============================================================
    fixed = gpt_fix_ocr(apikey, pil_img, meta)

    meta["invoice_no"] = fixed.get("invoice_no", meta["invoice_no"])
    meta["date"] = fixed.get("date", meta["date"])

    # ============================================================
    # Step 7：回傳結果
    # ============================================================
    return meta,  qr_left, qr_right



def gpt_fix_ocr(api_key, pil_img, raw_ocr):

    if not api_key:
        return raw_ocr

    client = OpenAI(api_key=api_key)

    # 轉成 base64
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = """
請從圖片中辨識台灣電子發票的三個欄位，並以 JSON 格式回覆：

{
  "invoice_no": "...",
  "date": "...",只要年月日，民國改西元
  "total_amount": "..."
}

務必只回傳純 JSON，不要加說明文字。
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ],
                }
            ],
        )

        reply = resp.choices[0].message.content

        # --- 修正：reply 可能是 list ---
        if isinstance(reply, list):
            text_part = ""
            for p in reply:
                if p.get("type") == "text":
                    text_part += p.get("text", "")
            reply = text_part

        # --- 確保 reply 是 JSON 字串 ---
        reply = reply.strip()
        start = reply.find("{")
        end = reply.rfind("}") + 1
        reply = reply[start:end]

        fixed = json.loads(reply)

        # --- 最終保險：確保三個欄位一定存在 ---
        return {
            "invoice_no": clean_invoice_no(fixed.get("invoice_no", "") or raw_ocr.get("invoice_no", "")),
            "date": fixed.get("date", "") or raw_ocr.get("date", ""),
            "total_amount": fixed.get("total_amount", "") or raw_ocr.get("total_amount", ""),
        }

    except Exception as e:
        st.error(f"GPT fallback 錯誤：{e}")
        return raw_ocr
    
def gpt_read_amount_from_roi(api_key: str, roi_img: Image.Image) -> str:
    if not api_key or roi_img is None:
        return "0"

    from openai import OpenAI
    import cv2
    import numpy as np
    import base64
    import io
    import re

    client = OpenAI(api_key=api_key)

    # ========= Step 1：保留原始細節，不做 dilate =========
    img = np.array(roi_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 各種版本
    _, th1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    inv1 = 255 - th1
    inv2 = 255 - th2

    candidates = [enhanced, th1, th2, inv1, inv2]
    best = candidates[np.argmin([np.mean(c) for c in candidates])]

    h, w = best.shape
    best_large = cv2.resize(best, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # ========= Step 2：轉 base64 給 GPT =========
    buf = io.BytesIO()
    Image.fromarray(best_large).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = """請讀出總金額，只回傳純數字。
只看冒號「:」右邊的第一組數字。
如果看起來像 39 請回 39；不要回推估的字。
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    ]
                }
            ],
            max_tokens=10,
            temperature=0.0
        )
        reply = response.choices[0].message.content.strip()

        # 先找 冒號後面的
        m = re.search(r'[:：]\s*(\d+)', reply)
        if m:
            return m.group(1)

        # fallback：純數字
        digits = re.sub(r"[^\d]", "", reply)
        if digits:
            return digits

    except:
        pass

    return "0"

# ------------------------------
# 最終穩定版：UNet  + GPT-4o-mini fallback
# ------------------------------


reader_invoice = easyocr.Reader(['en'], gpu=False)   # 專抓英文數字
reader_general = easyocr.Reader(['ch_tra','en'], gpu=False)


def ocr_easy(img):
    """
    img 可以是 PIL Image 或 numpy array
    EasyOCR 需要 numpy array (RGB)
    """
    # 如果是 PIL Image → 轉 numpy
    if isinstance(img, Image.Image):
        np_img = np.array(img.convert("RGB"))
    else:
        np_img = img

    # EasyOCR 讀取
    result = reader_invoice.readtext(np_img, detail=1)

    # 把辨識結果接起來
    text = "".join([r[1] for r in result])
    return text.strip()


def parse_invoice_date(date_crop):
    if not date_crop:
        return ""

    np_img = np.array(date_crop)
    raw_list = reader.readtext(np_img, detail=0)
    raw = "".join(raw_list)
    
    raw_clean = raw.replace("年", "-").replace("月", "-").replace("日", "")
    raw_clean = raw_clean.replace("/", "-").replace(".", "-").replace(" ", "")

    # 抓出所有數字
    nums = re.findall(r"\d+", raw_clean)

    # ----------------------------------------
    # 1) 民國年（3 位數）→ 西元
    # ----------------------------------------
    if len(nums) >= 3 and len(nums[0]) == 3:     # 例如 114-07-08
        y = int(nums[0]) + 1911
        m = int(nums[1])
        d = int(nums[2])
        return f"{y:04d}-{m:02d}-{d:02d}"

    # ----------------------------------------
    # 2) 西元年（4 位數，包含被 OCR 搞壞的）
    # ----------------------------------------
    m = re.search(r"(\d{4})[-]?(\d{1,2})[-]?(\d{1,2})", raw_clean)
    if m:
        y, mm, dd = map(int, m.groups())

        # ---------- 年份修復邏輯 ----------
        # 台灣電子發票年份落在 2010~2035
        if not (2010 <= y <= 2035):
            y_str = str(y)
            # 最強修復法：把「20」固定好
            y_str = "20" + y_str[2:]  # 2116 → 2016，2076 → 2076
            y = int(y_str)

            # 若仍不合理，強制拉回目前世代（2020~2026）
            if y < 2010 or y > 2035:
                y = 2020 + (y % 10)

        # 月/日修復（避免 23月 88日）
        mm = max(1, min(mm, 12))
        dd = max(1, min(dd, 31))

        return f"{y:04d}-{mm:02d}-{dd:02d}"

    return ""

# ============================================================
# 備援函數：當 QR 完全失效時，用 UNet + OCR 強行救回
# ============================================================
def extract_from_crops_ocr(crops: dict) -> dict:
    """
    V42 — 最終穩定金額 OCR（與 Debug 模式一致）
    整合發票號碼、日期、金額三區塊的純 OCR 備援
    """
    meta = {"invoice_no": "", "date": "", "total_amount": ""}

    # ================== 發票號碼 ==================
    inv_crop = crops.get("invoice_no")
    if inv_crop is not None:
        pad = 30
        np_img = cv2.copyMakeBorder(
            np.array(inv_crop),
            top=10, bottom=10,
            left=pad, right=pad + 20,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        result = reader.readtext(np_img, detail=1, 
                                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-—– ')
        texts = [r[1].upper() for r in result]
        raw_text = " ".join(texts)

        oracle_fix = str.maketrans({
            "亍":"7","丂":"7","丁":"7","了":"7","丄":"7",
            "工":"1","丨":"1","Ｏ":"O","０":"0",
            "－":"-","—":"-","–":"-"," ":""
        })
        text_fixed = raw_text.translate(oracle_fix)

        patterns = [
            r"[A-Z]{2}[\s—–-]*\d{8}",
            r"[A-Z]{2}\s*\d{8}",
            r"[A-Z]{2}\d{8}",
            r"\d{8}[A-Z]{2}",
        ]
        invoice_num = None
        for pat in patterns:
            m = re.search(pat, text_fixed)
            if m:
                clean = re.sub(r"[^A-Z0-9]", "", m.group(0))
                if len(clean) == 10 and clean[:2].isalpha() and clean[2:].isdigit():
                    invoice_num = clean
                    break

        if not invoice_num:
            heads = re.findall(r"[A-Z]{2}", text_fixed)
            head = heads[0] if heads else "XX"
            digits = "".join(re.findall(r"\d", text_fixed))
            if len(digits) >= 6:
                num_part = (digits[:8] + "77").ljust(8, "7")[:8]
                invoice_num = head + num_part

        if invoice_num:
            meta["invoice_no"] = invoice_num

    # ================== 日期 ==================
    date_crop = crops.get("date")
    if date_crop is not None:
        text = reader.readtext(np.array(date_crop), detail=0)
        raw = " ".join(text)

        cleaned = raw.upper()
        cleaned = cleaned.replace("O","0").replace("I","1").replace("C","0")\
                        .replace("S","5").replace("G","6").replace("Z","2")\
                        .replace("B","8").replace("o","0").replace(".","-")
        cleaned = re.sub(r"[^\d\-\/]", "", cleaned)

        patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            r"\d{7,8}",
            r"\d{2,3}[-/]\d{1,2}[-/]\d{1,2}",
        ]
        for p in patterns:
            m = re.search(p, cleaned)
            if m:
                dt = m.group(0).replace("/", "-")
                digits = dt.replace("-", "")
                if len(digits) == 7:
                    roc = int(digits[:3])
                    dt = f"{roc + 1911}-{digits[3:5]}-{digits[5:]}"
                meta["date"] = dt
                break

    # ================== 金額（無需 Tesseract 版本） ==================
        amount_crop = crops.get("total_amount")
        if amount_crop is not None:

            st.write("🟩 UNet 金額 ROI：")
            st.image(amount_crop, width=380)

            # ------- GPT 讀取 ROI 金額 -------
            gpt_roi_amount = gpt_read_amount_from_roi(apikey, amount_crop)

            st.write("🟩 GPT ROI 金額（raw）:", gpt_roi_amount)

            if gpt_roi_amount.isdigit():
                meta["total_amount"] = gpt_roi_amount
                # 不 return，仍讓後面 gpt_fix_ocr() 有機會修補其它欄位
            else:
                st.warning("GPT ROI 未成功 → 將使用 OCR/後處理 fallback。")
    return meta

# ------------------------------
# QR：pyzxing (主力)
# ------------------------------
def decode_qr_pyzxing(pil_img):
    """使用 pyzxing 解析整張圖片的所有 QR"""
    try:
        from pyzxing import BarCodeReader
        reader = BarCodeReader()
        
        # Save temp
        tmp = "tmp_qr.png"
        pil_img.save(tmp)

        result = reader.decode(tmp)
        if not result:
            return []

        decoded = []
        for r in result:
            if "raw" in r:
                # pyzxing 有 raw bytes → decode 成 utf-8
                try:
                    decoded.append(r["raw"].decode("utf-8"))
                except:
                    decoded.append(r["raw"].decode("big5", errors="ignore"))
            elif "text" in r:
                decoded.append(r["text"])
        return decoded
    except Exception:
        return []


# ------------------------------
# QR：OpenCV fallback
# ------------------------------
def decode_qr_opencv(pil_img):
    """OpenCV detectAndDecodeMulti 當備用方案"""
    try:
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        det = cv2.QRCodeDetector()
        ok, decoded_info, pts, _ = det.detectAndDecodeMulti(cv_img)

        if not ok:
            return []
        return [d for d in decoded_info if d]
    except:
        return []


# ------------------------------
# TEXT QR → 品項解析
# ------------------------------
import re

def parse_text_qr_items(text: str):
    if not text or not isinstance(text, str):
        return []

    # Step 1：載具+贈品移除（通殺 4:0 / 5:0 / 9:0 + 孤立1）
    text = re.sub(r'^[A-Z0-9+/=\s※\*\-:]*?\*{5,}.*?[:：]\d+[:：]0[:：](1)?', '', text, flags=re.DOTALL)
    text = re.sub(r'^[※\*\s:-]+', '', text)

    # Step 2：正規化
    clean = re.sub(r'[\*＊\s　@＠$＄:：]+', '|', text.strip())
    clean = re.sub(r'^\|+', '', clean)
    clean = re.sub(r'\|+', '|', clean)

    parts = [p.strip() for p in clean.split('|') if p.strip()]

    # 用字典做「同品名+同單價」合併
    item_dict = {}

    i = 0
    while i + 2 < len(parts):
        try:
            qty = float(parts[i + 1])
            price = float(parts[i + 2])
            if price <= 0 or qty <= 0 or qty > 1000 or price > 200000:
                i += 1
                continue
        except:
            i += 1
            continue

        # 品名往前吃
        name_parts = []
        j = i
        while j >= 0:
            part = parts[j]
            if part == "1" and j == 0:  # 最前面的孤立1直接丟
                j -= 1
                continue
            if re.fullmatch(r'\d+\.?\d*', part):
                break
            name_parts.insert(0, part)
            j -= 1

        name = ''.join(name_parts).strip(" :：*＊@＄.、，,()（）-－")

        # 最後防線：如果品名以1開頭 + 第二個字是中文 → 砍掉1
        if name and len(name) > 1 and name[0] == "1" and "\u4e00" <= name[1] <= "\u9fff":
            name = name[1:]

        if not name or len(name) > 40 or any(kw in name for kw in ["總計","小計","稅","載具","點","贈","紅利","折扣"]):
            i += 3
            continue

        # 合併邏輯：同品名 + 同單價 → 數量相加
        key = (name, price)
        if key in item_dict:
            item_dict[key]["qty"] += qty
            item_dict[key]["amount"] = round(item_dict[key]["qty"] * price, 2)
        else:
            item_dict[key] = {
                "name": name,
                "qty": qty,
                "price": price,
                "amount": round(qty * price, 2)
            }

        i += 3

    # 轉回 list
    final_items = list(item_dict.values())

    # 按金額從大到小排序（好看）
    final_items.sort(key=lambda x: x["amount"], reverse=True)

    return final_items
# ------------------------------
# 品項 → 金額等比例調整（符合總金額）
# ------------------------------
def adjust_items_with_total(items, total_amount):
    """
    將 TEXT QR 品項以比例調整，並四捨五入到整數，
    最後用差額補到最大金額的品項，確保總金額完全對齊。
    """

    if not items or total_amount is None:
        return items

    try:
        total_amount = int(float(total_amount))
    except:
        return items

    # 1. 計算原始小計
    original_subtotal = sum(it["qty"] * it["price"] for it in items)
    if original_subtotal <= 0:
        return items

    ratio = total_amount / original_subtotal

    # 2. 按比例 + 四捨五入
    adjusted = []
    for it in items:
        new_price = it["price"] * ratio
        new_amount = round(new_price * it["qty"])  # ← 四捨五入整數
        adjusted.append({
            "name": it["name"],
            "qty": it["qty"],
            "price": round(new_price),  # 單價四捨五入
            "amount": new_amount,
        })

    # 3. 檢查與總金額誤差
    sum_after = sum(it["amount"] for it in adjusted)
    diff = total_amount - sum_after

    # 4. 用最大 amount 的品項補差額（避免不自然）
    if diff != 0:
        idx = max(range(len(adjusted)), key=lambda i: adjusted[i]["amount"])
        adjusted[idx]["amount"] += diff

    return adjusted


# ------------------------------
# 主流程：全圖偵測 → 合併 TEXT QR → 解析 → 回傳
# ------------------------------
import re

def is_real_text_qr(text: str) -> bool:
    if not text:
        return False

    text = text.strip()

    # ------ 排除新版主 QR ------
    if text.startswith(("QF", "QG", "QA", "QS")):
        return False

    # ------ 排除舊版主 QR ------
    if text.startswith("**") and re.match(r"\*\*[A-Z]{2}\d{8}", text):
        return False

    # ------ 規律 1：中文 + 數量 + 價格
    if re.search(r"[\u4E00-\u9FFF]+.*:\d+:\d+", text):
        return True

    # ------ 規律 2：至少兩個冒號（品項格式）------
    if text.count(":") >= 2:
        return True

    return False

def debug_qr_classification(text: str):
    """
    新版：優先判斷 TEXT QR，即使是 QG / QF 開頭也要檢查是否含品項。
    """
    if not text:
        return False, "EMPTY"

    t = text.strip()

    # 🔥 1. 優先判斷是否為 TEXT QR
    # 有中文品名 + 冒號 + 數量 + 價格
    if re.search(r"[\u4E00-\u9FFF].*:\d+:\d+", t):
        return True, "TEXT:中文+數量+價格"

    # 或者至少兩個冒號，也視為 TEXT 格式
    if t.count(":") >= 2 and re.search(r"[\u4E00-\u9FFF]", t):
        return True, "TEXT:多冒號+中文"

    # 🔥 2. 才判斷是否為新版主 QR（QG/QF/etc）
    if t.startswith(("QF", "QG", "QA", "QS")):
        return False, "主QR:新版v3"

    # 舊版主 QR
    if t.startswith("**") and re.match(r"\*\*[A-Z]{2}\d{8}", t):
        return False, "主QR:舊版"

    return False, "NOT_TEXT"



def detect_invoice_items_from_qr(qr_left, qr_right, total_amount):
    """
    直接使用 parse_qr_invoice() 的輸出
    不重新掃整張圖（pyzxing / opencv）
    TEXT QR Debugger 永遠不會空
    """

    st.markdown("### 🐞 TEXT QR Debugger（from parse_qr_invoice）")

    # Step 1：把前面抓到的 QR 收進來
    raw_all = []
    if qr_left:
        raw_all.append(qr_left)
    if qr_right:
        raw_all.append(qr_right)

    st.write("📌 raw_all (parse_qr_invoice 結果)")
    st.write(raw_all)

    # Step 2：分類
    main_qr = []
    text_qr = []
    debug_details = []

    for raw in raw_all:
        is_text, rule = debug_qr_classification(raw)
        debug_details.append((raw, rule))

        if rule.startswith("主QR"):
            main_qr.append(raw)
        elif is_text:
            text_qr.append(raw)

    st.write("📌 主 QR 分類結果：", main_qr)
    st.write("📌 TEXT QR 分類結果：", text_qr)
    st.write("📌 Rule 判斷：")
    for raw, rule in debug_details:
        st.write(f"- `{raw}` → `{rule}`")

    # Step 3：沒有 TEXT QR → 結束
    if not text_qr:
        st.warning("⚠ 未偵測到 TEXT QR")
        return {
            "raw_all": raw_all,
            "main_qr": main_qr,
            "text_qr": [],
            "debug": debug_details
        }, []

    # Step 4：合併 TEXT QR
    combined_text = ":".join(text_qr)
    st.write("📌 合併後 TEXT QR：")
    st.code(combined_text)

    # Step 5：解析 items
    items = parse_text_qr_items(combined_text)
    st.write("📌 解析後 items：")
    st.write(items)

    if not items:
        st.error("❌ parse_text_qr_items 回傳空（格式怪異）")
        return {
            "raw_all": raw_all,
            "main_qr": main_qr,
            "text_qr": text_qr,
            "combined_text": combined_text,
            "debug": debug_details
        }, []

    # Step 6：金額等比例調整
    items = adjust_items_with_total(items, total_amount)

    return {
        "raw_all": raw_all,
        "main_qr": main_qr,
        "text_qr": text_qr,
        "combined_text": combined_text,
        "debug": debug_details
    }, items

# ============================================================
# Part 4 — UI + Supabase 儲存 + Tab1 / Tab2 主體
# ============================================================
# ============================================================
# 儲存發票（主檔）
# ============================================================
def save_invoice_main(meta, total_amount, category, note):
    """回傳 invoice_id 或 None"""
    try:
        data = {
            "invoice_no": meta.get("invoice_no", ""),
            "date": meta.get("date", None),
            "total_amount": float(total_amount),
            "category": category,
            "note": note,
        }
        res = supabase.table("invoices_data").insert(data).execute()
        if res.data:
            return res.data[0]["id"]
        return None
    except Exception as e:
        st.error(f"❌ 儲存發票主檔失敗：{e}")
        return None


# ============================================================
# 儲存品項（子檔）
# ============================================================
def save_invoice_items(invoice_id, items):
    try:
        rows = []
        for it in items:
            rows.append({
                "invoice_id": invoice_id,
                "name": it["name"],
                "qty": it["qty"],
                "price": it["price"],
                "amount": it["amount"],
            })

        supabase.table("invoice_items").insert(rows).execute()
        return True
    except Exception as e:
        st.error(f"❌ 儲存品項失敗：{e}")
        return False


# ============================================================
# Tab Layout
# ============================================================
tab1, tab2 = st.tabs(["📤 發票上傳", "📊 發票分析儀表板"])

with tab1:

    st.markdown("<h2>📤 上傳並辨識發票</h2>", unsafe_allow_html=True)

    uploaded = st.file_uploader("請選擇發票圖片 (JPG / PNG)", type=["jpg", "jpeg", "png"])

    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/best_unet_model.pth")

    # ==============================
    # 🔹 Case A：沒有重新上傳 → 使用上一次的結果
    # ==============================
    if not uploaded and "last_meta" in st.session_state:

        pil_img = st.session_state["last_image"]
        meta = st.session_state["last_meta"]
        items = st.session_state["last_items"]

        st.image(pil_img, caption="📸 原始影像 (快取)", width='stretch')

        st.markdown("### 🧾 發票資訊（已快取，不重新辨識）")
        st.write(f"**發票號碼：** {meta['invoice_no']}")
        st.write(f"**日期：** {meta['date']}")
        st.write(f"**總金額：** NT$ {meta['total_amount']}")

    # ==============================
    # 🔹 Case B：使用者有上傳 → 重新辨識
    # ==============================
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")

        col_img, col_info = st.columns([1, 1])

        with col_img:
            st.image(pil_img, caption="📸 原始影像", width='stretch')

        with col_info:
            meta, qr_left, qr_right = extract_invoice_meta(
                uploaded_file=uploaded,   
                pil_img=pil_img,
                checkpoint_path=checkpoint_path,
                apikey=apikey
            )

            meta = meta or {}
            # ===== 儲存結果（避免 Rerun 重跑辨識）=====
            st.session_state["last_image"] = pil_img
            st.session_state["last_meta"] = meta

            st.markdown("### 🧾 發票資訊")
            st.write(f"**發票號碼：** {meta.get('invoice_no', '未知')}")
            st.write(f"**日期：** {meta.get('date', '未知')}")
            st.write(f"**總金額：** NT$ {meta.get('total_amount', '未知')}")

        # ==============================
        # 🔍 QR Code 掃描
        # ==============================
        with st.spinner("📡 TEXT QR 掃描中…"):
    
            debug_info, items = detect_invoice_items_from_qr(
                qr_left,
                qr_right,
                meta.get("total_amount", "0")
            )
            
        st.session_state["last_items"] = items

    # ==============================
    # 📦 TEXT QR 品項顯示
    # ==============================
    st.markdown("### 📦 TEXT QR 品項")

    if "last_items" in st.session_state:
        items = st.session_state["last_items"]

        if items:
            df_items = pd.DataFrame(items)

            df_items["price"] = df_items["price"].astype(float).round(0)
            df_items["qty"] = df_items["qty"].astype(float)

            # 🔥 合併同品項
            df_items = (
                df_items.groupby("name", as_index=False)
                .agg({"qty": "sum", "price": "first"})
            )

            df_items["amount"] = (df_items["qty"] * df_items["price"]).round(0)

            st.dataframe(df_items, width='stretch')
        else:
            st.info("📭 未偵測到 TEXT QR 品項")

    # ==============================
    # 🏷 類別 + 備註
    # ==============================
    st.markdown("### 🏷 類別與備註")
    category = st.selectbox("類別 Category", ["餐飲","購物","交通","娛樂","日用品","其他"])
    note = st.text_input("備註 Note")

    # ============================================================
    # 🟩 背景儲存功能（不阻塞、不卡畫面）
    # ============================================================
    import threading

    def async_save_invoice(meta, total_amount, category, note, items):
        def job():
            try:
                st.session_state.save_status = "saving"
                st.session_state.last_save_time = None

                # 儲存主表
                res = supabase.table("invoices_data").insert({
                    "invoice_no": meta.get("invoice_no", "未知"),
                    "date": meta.get("date"),
                    "total_amount": float(total_amount),
                    "category": category,
                    "note": note or None,
                }).execute()

                if not res.data:
                    raise Exception("主表儲存失敗")

                invoice_id = res.data[0]["id"]

                # 批次儲存品項（超快）
                if items:
                    batch = []
                    for it in items:
                        batch.append({
                            "invoice_id": invoice_id,
                            "name": str(it["name"]),
                            "qty": float(it["qty"]),
                            "price": float(it["price"]),
                            "amount": float(it["amount"]),
                        })
                    supabase.table("invoice_items").insert(batch).execute()

                # 成功！
                st.session_state.save_status = "success"
                st.session_state.last_save_time = pd.Timestamp.now().strftime("%H:%M:%S")

            except Exception as e:
                st.session_state.save_status = "error"
                st.session_state.last_error = str(e)

        threading.Thread(target=job, daemon=True).start()

    # ============================================================
    # 💾 儲存按鈕（不卡畫面，不重跑辨識）
    # ============================================================
    if supabase:
        col_save1, col_save2 = st.columns([1, 5])
        with col_save1:
            # 關鍵防呆：正在儲存時按鈕變灰 + 不能再按
            save_button_disabled = (st.session_state.save_status == "saving")
            
            if st.button(
                "儲存" if not save_button_disabled else "儲存中…",
                type="primary",
                use_container_width=True,
                disabled=save_button_disabled,   # 這行是王道！
                key="save_btn"
            ):
                try:
                    total_amount = float(re.sub(r"[^\d.]", "", str(meta.get("total_amount", "0"))))
                except:
                    total_amount = 0.0
                    
                async_save_invoice(meta, total_amount, category, note, items)
                # 按下去就立刻改狀態（避免狂按）
                st.session_state.save_status = "saving"

        # === 即時狀態通知（保持不變）===
        status = st.session_state.save_status
        
        if status == "saving":
            st.info("正在背景儲存中… 你可以馬上辨識下一張！")
            
        elif status == "success":
            st.success(f"儲存成功！（{st.session_state.last_save_time}）")
            st.balloons()
            time.sleep(2.5)
            st.session_state.save_status = "idle"
            st.rerun()
            
        elif status == "error":
            st.error(f"儲存失敗：{st.session_state.last_error}")
            if st.button("重試儲存"):
                st.session_state.save_status = "idle"
                st.rerun()
                
        else:
            st.info("可以開始儲存下一張發票了喔～")   # 改得更清楚！
# ============================================================
# TAB 2 — 儀表板（使用 cache，完全不會拖慢 TAB1）
# ============================================================

# --------- 🚀 加速：Supabase 讀取快取 --------------
@st.cache_data(ttl=300, show_spinner=False)  # 5分鐘內絕對不重抓
def load_all_data():
    try:
        # 一次把主表 + 所有品項一起抓下來（Supabase 支援 nested select）
        response = supabase.table("invoices_data")\
            .select("*, invoice_items(*)", count="exact")\
            .order("date", desc=True)\
            .execute()
        
        data = response.data
        # 把嵌套的 invoice_items 展開成平的（方便後面使用）
        flat_rows = []
        for inv in data:
            items = inv.pop("invoice_items", [])
            if not items:
                flat_rows.append(inv)
            else:
                for item in items:
                    row = inv.copy()
                    row.update(item)
                    flat_rows.append(row)
        return pd.DataFrame(flat_rows)
    except Exception as e:
        st.error(f"載入資料失敗：{e}")
        return pd.DataFrame()


# --------- 🚀 加速：圖表快取 ---------------------
@st.cache_resource
def plot_monthly(df_inv):
    monthly = df_inv.groupby("year_month")["total_amount"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    return monthly


with tab2:
    st.markdown("<h2>發票記帳儀表板</h2>", unsafe_allow_html=True)

    if not supabase:
        st.warning("Supabase 未連線")
        st.stop()

    # ========= 超快載入：一次抓全部資料 + 5分鐘快取 =========
    @st.cache_data(ttl=300, show_spinner=False)  # 5分鐘快取
    def load_all_data():
        try:
            # Step 1: 抓主表
            inv_resp = supabase.table("invoices_data")\
                .select("*")\
                .order("date", desc=True)\
                .execute()
            
            if not inv_resp.data:
                return pd.DataFrame()

            df_inv = pd.DataFrame(inv_resp.data)

            # Step 2: 抓品項表
            items_resp = supabase.table("invoice_items")\
                .select("*")\
                .execute()

            if not items_resp.data:
                # 沒有品項也沒關係，至少主表有資料
                df_inv["name"] = None
                df_inv["qty"] = None
                df_inv["price"] = None
                df_inv["amount"] = None
                return df_inv

            df_items = pd.DataFrame(items_resp.data)

            # Step 3: 合併（左外連結）
            df_merged = df_inv.merge(df_items, left_on="id", right_on="invoice_id", how="left", suffixes=("", "_item"))

            return df_merged

        except Exception as e:
            st.error(f"載入資料失敗：{e}")
            return pd.DataFrame()
        

    df_all = load_all_data()

    if df_all.empty:
        st.info("還沒有任何發票資料，快去上傳第一張吧！")
        st.stop()

    # 預處理日期
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["year_month"] = df_all["date"].dt.to_period("M").astype(str)

    # ========= KPI =========
    col1, col2, col3 = st.columns(3)
    current_month_str = df_all["year_month"].max()
    df_current = df_all[df_all["year_month"] == current_month_str]

    with col1:
        st.metric("本月消費", f"NT$ {df_current['total_amount'].sum():,.0f}")

    with col2:
        months = sorted(df_all["year_month"].unique(), reverse=True)
        last_month_str = months[1] if len(months) > 1 else current_month_str
        last_amount = df_all[df_all["year_month"] == last_month_str]["total_amount"].sum()
        growth = ((df_current["total_amount"].sum() - last_amount) / last_amount * 100) if last_amount > 0 else 0
        st.metric("月成長率", f"{growth:+.1f}%")

    with col3:
        top_cat = df_current.groupby("category")["total_amount"].sum()
        st.metric("最大類別", top_cat.idxmax() if not top_cat.empty else "無")

    # ========= 每月支出趨勢 =========
    monthly = df_all.groupby("year_month")["total_amount"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].astype(str)
    st.line_chart(monthly.set_index("year_month"))

    # ========= 類別圓餅圖 =========
    cat_sum = df_all.groupby("category")["total_amount"].sum()
    if not cat_sum.empty:
        fig = px.pie(values=cat_sum.values, names=cat_sum.index, hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

    # ========= 選擇月份 =========
    months = sorted(df_all["year_month"].unique(), reverse=True)
    selected_month = st.selectbox("查看特定月份", months, index=0)
    df_month = df_all[df_all["year_month"] == selected_month]

    # 顯示該月發票列表
    display_cols = ["date", "invoice_no", "total_amount", "category", "note"]
    st.dataframe(
        df_month[display_cols].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # ========= 選擇發票查看品項 =========
    invoice_ids = df_month["id"].dropna().unique().tolist()
    if invoice_ids:
        selected_id = st.selectbox(
            "選擇發票查看品項",
            options=invoice_ids,
            format_func=lambda x: f"{df_month[df_month['id']==x]['date'].iloc[0].strftime('%Y-%m-%d')}｜{df_month[df_month['id']==x]['invoice_no'].iloc[0]}｜NT${df_month[df_month['id']==x]['total_amount'].iloc[0]:,.0f}"
        )

        items_df = df_month[df_month["id"] == selected_id]
        if "name" in items_df.columns and not items_df["name"].isna().all():
            st.dataframe(items_df[["name", "qty", "price", "amount"]], use_container_width=True)
        else:
            st.info("這張發票沒有品項資料（可能是用 QR 直接存的）")

    # ========= 刪除發票功能 =========
    st.markdown("---")
    st.markdown("### 刪除發票（含所有品項）")

    if invoice_ids:
        delete_id = st.selectbox(
            "選擇要刪除的發票（小心！無法復原）",
            options=invoice_ids,
            format_func=lambda x: f"{df_month[df_month['id']==x]['date'].iloc[0].strftime('%Y-%m-%d')} | {df_month[df_month['id']==x]['invoice_no'].iloc[0]} | NT${df_month[df_month['id']==x]['total_amount'].iloc[0]:,.0f}",
            key="delete_select"
        )

        col_del1, col_del2 = st.columns([1, 4])
        with col_del1:
            if st.button("🗑 刪除這張發票（不可恢復）", type="secondary", use_container_width=True):
                with st.spinner("刪除中…"):
                    try:
                        # 真的刪除
                        supabase.table("invoices_data").delete().eq("id", delete_id).execute()
                        
                        # 強制清除快取 ← 這一行是王道！
                        st.cache_data.clear()
                        
                        st.success("已成功刪除！畫面即將更新")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()  # 重新載入最新資料
                    except Exception as e:
                        st.error(f"刪除失敗：{e}")
