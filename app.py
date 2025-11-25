# ============================================================
# app.py — v15.4 (修正儀表板筆數跳動問題：強制日期標準化)
# ============================================================

import os
import io
import re
from datetime import datetime
import time
import json 
import base64 

import streamlit as st
from PIL import Image
import psycopg2
import pandas as pd
import pytesseract
import numpy as np
import plotly.express as px

# 導入 UNet 相關
try:
    from inference import run_unet_inference
    from inference import visualize_mask
except ImportError:
    # 如果 inference 模組不存在，提供空函式避免程式崩潰
    def run_unet_inference(pil_img, checkpoint_path):
        # 回傳 None, None, Empty dict
        return None, None, {} 
    def visualize_mask(mask):
        return Image.new('RGB', (100, 100), color = 'red')

from openai import OpenAI

# ------------------------------------------------------------
# 1. 自動偵測 Tesseract.exe（Windows）
# ------------------------------------------------------------
def auto_set_tesseract_path():
    """自動偵測 Tesseract OCR 執行檔路徑"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\user\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            return p
    return None

TESSERACT_PATH = auto_set_tesseract_path()


# ------------------------------------------------------------
# 2. PostgreSQL 設定
# ------------------------------------------------------------
def get_db_conn():
    """獲取一個新的資料庫連線"""
    try:
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            user="postgres",
            password="postgres",
            dbname="invoices_db",
        )
        return conn
    except psycopg2.Error as e:
        st.error(f"資料庫連線失敗: {e}")
        return None

# ------------------------------------------------------------
# 3. OpenAI 配置
# ------------------------------------------------------------
# 🚨 修正：請在這裡填入您有效的 API Key
client = OpenAI(api_key=" ")

# ------------------------------------------------------------
# 4. 常數
# ------------------------------------------------------------
CATEGORIES = ["餐飲", "交通", "購物", "娛樂", "醫療", "教育", "雜項", "收入"]
CHECKPOINT_PATH = "checkpoints/unet_epoch30.pth" # 假設您的模型在這裡

# ------------------------------------------------------------
# 5. 函數：LLM 驗證與修正 (V15.4 修正重點：強制 ISO 日期格式)
# ------------------------------------------------------------

def llm_validate_and_correct(img_bytes, ocr_results, user_query):
    """使用 GPT-4-Vision 進行 OCR 結果驗證與修正"""
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    # 🌟 V15.4 修正點：在 Prompt 中明確要求 YYYY-MM-DD 格式
    prompt = f"""
    您是一位專業的發票資料審核員。您面前有一張發票圖片和初步的 OCR 辨識結果。
    
    **OCR 結果:**
    發票號碼: {ocr_results.get('invoice_no', 'N/A')}
    日期: {ocr_results.get('date', 'N/A')}
    金額: {ocr_results.get('total_amount', 'N/A')}
    
    **任務:**
    1. **檢查**圖片，特別是 OCR 辨識出來的**發票號碼**、**日期**和**總金額**是否正確。
    2. **修正**任何錯誤，並以 **JSON** 格式回傳最終結果。JSON 必須包含 "發票號碼"、"日期" 和 "金額" 三個鍵。
       - **日期** 必須使用 ISO 8601 標準格式 `YYYY-MM-DD`，例如 `2024-06-25`。
       - **金額** 必須是純數字，例如 `1250`。
    3. 如果某個欄位無法辨識，請填寫 `"N/A"`。
    
    **用戶額外請求:** {user_query}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        llm_output = json.loads(response.choices[0].message.content)
        return llm_output
    except Exception as e:
        print(f"LLM 呼叫或解析錯誤: {e}")
        return None


# ------------------------------------------------------------
# 6. 函數：資料儲存
# ------------------------------------------------------------
def save_invoice(img_bytes, data):
    conn = get_db_conn()
    if not conn: return False 
    
    cur = conn.cursor()
    img_binary = psycopg2.Binary(img_bytes)

    try:
        # 1. 插入主要發票紀錄
        cur.execute(
            """
            INSERT INTO invoices (invoice_image, created_at)
            VALUES (%s, NOW()) RETURNING id; 
            """,
            (img_binary,),
        )
        invoice_id = cur.fetchone()[0]

        # 2. 插入欄位資料 (包含備註)
        data_to_save = {**data, "note": data.get("note", "無")} 
        
        for k, v in data_to_save.items():
            if k == 'note' and v == "無": continue 
            
            cur.execute(
                """
                INSERT INTO invoice_fields (invoice_id, field_name, field_value)
                VALUES (%s,%s,%s)
                """,
                (invoice_id, k, str(v)),
            )

        conn.commit()
        st.success(f"✔ 資料已寫入資料庫，Invoice ID={invoice_id}")
        return True 

    except psycopg2.Error as e:
        st.error(f"寫入資料庫失敗: {e}")
        conn.rollback()
        return False 

    finally:
        # 確保游標和連線關閉
        if cur: cur.close()
        if conn: conn.close()


# ------------------------------------------------------------
# 7. 函數：資料查詢 (專用於儀表板)
# ------------------------------------------------------------
# 保持 @st.cache_data 啟用，但讓 save_invoice 負責清除它
@st.cache_data(ttl=600) 
def load_data_for_dashboard():
    conn = get_db_conn()
    if not conn: return pd.DataFrame()

    query = """
    SELECT 
        i.id, 
        i.created_at, 
        f_date.field_value AS date,
        f_amount.field_value AS total_amount,
        f_category.field_value AS category,
        f_invno.field_value AS invoice_no,
        f_note.field_value AS note -- 備註欄位
    FROM invoices i
    JOIN invoice_fields f_date ON i.id = f_date.invoice_id AND f_date.field_name = 'date'
    JOIN invoice_fields f_amount ON i.id = f_amount.invoice_id AND f_amount.field_name = 'total_amount'
    JOIN invoice_fields f_category ON i.id = f_category.invoice_id AND f_category.field_name = 'category'
    JOIN invoice_fields f_invno ON i.id = f_invno.invoice_id AND f_invno.field_name = 'invoice_no'
    LEFT JOIN invoice_fields f_note ON i.id = f_note.invoice_id AND f_note.field_name = 'note'
    ORDER BY i.created_at DESC;
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"資料庫讀取失敗: {e}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()
    
    if len(df) > 0:
        # 將數據轉換為正確的格式
        # 由於 LLM 已經被強制輸出 ISO 格式，這裡的轉換成功率會極高
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        # 關鍵過濾：丟棄任何轉換失敗的數據（例如日期或金額是 N/A 的紀錄）
        df = df.dropna(subset=['date', 'total_amount'])
        df['YearMonth'] = df['date'].dt.to_period('M')
        
    return df

# ------------------------------------------------------------
# 8. Streamlit 主體
# ------------------------------------------------------------

st.set_page_config(
    page_title="智能發票記帳神器",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 智能發票記帳神器")

# API Key 側邊欄輸入 
with st.sidebar:
    st.header("🔑 偵測與配置")
    openai_key = st.text_input("OpenAI API Key (gpt-4o)", type="password", help="用於 LLM 驗證與修正")
    
    if openai_key:
        client.api_key = openai_key
        st.success("OpenAI Key 已配置")
    else:
        st.warning("請在側邊欄輸入您的 OpenAI Key")
    
    if TESSERACT_PATH:
        st.info(f"Tesseract OCR 已偵測: {TESSERACT_PATH}")
    else:
        st.error("Tesseract OCR 未偵測到。請檢查路徑或安裝。")

# --- Tabs ---
tab1, tab2 = st.tabs(["🧾 掃描與記錄", "📊 分析與紀錄"])

# ========== TAB 1：掃描與記錄 ==========
with tab1:
    st.header("發票掃描與 AI 辨識")

    col1_upload, col2_control = st.columns([1, 2])
    
    uploaded = col1_upload.file_uploader(
        "**請上傳發票圖片 (JPG/PNG)**", 
        type=["jpg", "png", "jpeg"],
        help="建議圖片清晰、對焦良好"
    )

    # 執行辨識按鈕
    if 'processing' not in st.session_state:
        st.session_state.processing = False
        
    # 狀態初始化
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {
            "inv_no": "N/A",
            "parsed_date": "N/A",
            "amount": 0,
            "pil_img": None # 初始為 None
        }
        
    # V15.2 修正點：使用一個額外的 state 來追蹤檔案的 hash，避免無限循環
    if 'last_uploaded_hash' not in st.session_state:
        st.session_state.last_uploaded_hash = None
        
    current_uploaded_hash = None
    if uploaded is not None:
        # 簡易 hash 計算，判斷是否為新的檔案
        current_uploaded_hash = hash(uploaded.getvalue()) 
        
        # 邏輯修正：如果當前檔案的 hash 與上次處理的 hash 不一樣 (代表新檔案上傳)
        # 並且 last_uploaded_hash 已經被設定過 (避免第一次進入時就重跑)
        if current_uploaded_hash != st.session_state.last_uploaded_hash and st.session_state.last_uploaded_hash is not None:
            # 清理 current_data
            st.session_state.current_data = {
                "inv_no": "N/A", "parsed_date": "N/A", "amount": 0, "pil_img": None
            }
            # 更新 last_uploaded_hash
            st.session_state.last_uploaded_hash = current_uploaded_hash
            st.rerun() # 刷新頁面以清除舊預覽

    # 確保第一次上傳時 last_uploaded_hash 被設定
    if uploaded is not None and st.session_state.last_uploaded_hash is None:
        st.session_state.last_uploaded_hash = hash(uploaded.getvalue())


    process_button = col2_control.button(
        "🧠 啟動 AI 辨識", 
        type="secondary",
        disabled=uploaded is None or st.session_state.processing
    )
    
    # 處理流程只有在按下按鈕且檔案存在時才啟動
    if uploaded and process_button:
        st.session_state.processing = True
        
        with st.spinner("🚀 AI 辨識中 (UNet Segmentation -> Tesseract OCR -> GPT-4o 驗證)..."):
            
            # --- 影像載入與準備 ---
            img_bytes = uploaded.getvalue()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # --- 1. UNet Segmentation + Bounding Box ---
            try:
                # 這裡假設 run_unet_inference 能夠正常運行
                mask, bboxes, crops_map = run_unet_inference(pil_img, CHECKPOINT_PATH)
            except Exception as e:
                # UNet 推論失敗時，仍然允許進入下一步，但 crops_map 可能是空的
                st.error(f"UNet 推論失敗: {e}")
                crops_map = {} 
                
            
            # --- 2. Tesseract OCR ---
            ocr_results = {}
            for field, cropped_img in crops_map.items():
                if cropped_img:
                    # 假設這裡 Tesseract OCR 執行
                    # 修正點: 清理發票號碼中的破折號
                    ocr_text = pytesseract.image_to_string(cropped_img, lang='eng', config='--psm 6').strip()
                    ocr_results[field] = ocr_text.replace('\n', ' ')
            
            # --- 3. LLM 驗證 ---
            if openai_key:
                # 傳遞額外指令，確保日期和金額標準化
                llm_output = llm_validate_and_correct(img_bytes, ocr_results, "請確保日期為 YYYY-MM-DD 格式，且總金額為純數字")
                
                if llm_output:
                    # 發票號碼清理 (移除中線)
                    raw_inv_no = llm_output.get("發票號碼", "N/A")
                    if isinstance(raw_inv_no, str):
                        inv_no = raw_inv_no.replace('-', '').strip() 
                    else:
                        inv_no = "N/A"
                    
                    # 這裡的 parsed_date 應該已經是 YYYY-MM-DD 格式
                    parsed_date = llm_output.get("日期", "N/A") 
                    
                    amount_str = str(llm_output.get("金額", "0")).replace(',', '').strip()
                    try:
                        # 移除所有非數字和小數點的字元
                        amount = float(re.sub(r'[^\d.]', '', amount_str))
                    except ValueError:
                        amount = "N/A"
                else:
                    st.error("LLM 驗證失敗，請手動修正資料。")
                    inv_no, parsed_date, amount = "N/A", "N/A", "N/A" 
            else:
                # 無 Key 狀態下，使用基礎 OCR 結果 (這裡仍可能產生格式問題)
                raw_inv_no = ocr_results.get('invoice_no', 'N/A')
                if isinstance(raw_inv_no, str):
                    inv_no = raw_inv_no.replace('-', '').strip()
                else:
                    inv_no = "N/A"
                    
                parsed_date = ocr_results.get('date', 'N/A')
                amount_str = ocr_results.get('total_amount', '0').replace(',', '').strip()
                try:
                    amount = float(re.sub(r'[^\d.]', '', amount_str))
                except ValueError:
                    amount = "N/A"
            
            st.session_state.processing = False
            # 儲存結果到 session_state
            st.session_state.current_data = {
                "inv_no": inv_no,
                "parsed_date": parsed_date,
                "amount": amount,
                "pil_img": pil_img
            }
            
            # 重新運行以顯示結果
            st.rerun()

    # --- 顯示圖片與結果 ---
    # 只要上傳了圖片，或者 session_state 中有圖片數據，就進入顯示區塊
    if uploaded or st.session_state.current_data["pil_img"] is not None:
        
        # 確保當 uploaded 存在但 current_data["pil_img"] 為 None 時，使用 uploaded 的圖片
        if uploaded and st.session_state.current_data["pil_img"] is None:
            img_bytes = uploaded.getvalue()
            st.session_state.current_data["pil_img"] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 從 session_state 讀取資料
        inv_no = st.session_state.current_data["inv_no"]
        parsed_date = st.session_state.current_data["parsed_date"]
        amount = st.session_state.current_data["amount"]
        pil_img = st.session_state.current_data["pil_img"]
        
        
        # UI 分割
        col1_img, col2_input = st.columns([3, 2])
        
        with col1_img:
            st.subheader("🖼️ 發票圖片預覽")
            st.image(pil_img, caption="原始發票圖片", use_container_width=True) 

        with col2_input:
            st.subheader("📝 確認與分類")
            
            # --- 辨識結果 ---
            st.metric("發票號碼", inv_no)
            st.metric("日期", parsed_date)

            # 手動修正金額
            current_amount = amount if isinstance(amount, (int, float)) else 0
            
            st.metric("AI 辨識金額", f"NT${current_amount:,.0f}" if isinstance(amount, (int, float)) else str(amount))
            
            final_amount = st.number_input(
                "手動修正金額", 
                min_value=0, 
                max_value=500000, 
                value=int(current_amount),
                step=1
            )

            # --- 分類與儲存控制 ---
            st.markdown("---")
            category = st.selectbox("消費類別", CATEGORIES)
            note = st.text_input("項目/備註", "")

            # ===== 儲存按鈕 =====
            is_valid = (
                isinstance(inv_no, str) and inv_no != "N/A" and 
                isinstance(parsed_date, str) and parsed_date != "N/A" and 
                final_amount > 0
            )
            
            if st.button("💾 確認儲存至資料庫", type="primary", disabled=not is_valid):
                if not is_valid:
                    st.error("資料無效 (發票號碼/日期/金額)，無法儲存。")
                else:
                    data = {
                        "invoice_no": inv_no,
                        "date": parsed_date, # 這裡的日期必須是標準格式
                        "total_amount": final_amount,
                        "category": category,
                        "note": note
                    }
                    
                    img_bytes_io = io.BytesIO()
                    # 確保圖片存在才能儲存
                    if st.session_state.current_data["pil_img"]:
                        st.session_state.current_data["pil_img"].save(img_bytes_io, format='JPEG')
                        img_to_save = img_bytes_io.getvalue()
                    else:
                        img_to_save = b''
                    
                    # 執行儲存並接收結果 
                    save_success = save_invoice(img_to_save, data)
                    
                    if save_success:
                        # V15.3 關鍵修正點：儲存成功時清除緩存
                        # 確保下次載入儀表板時會重新查詢資料庫
                        st.cache_data.clear() 
                        
                    # 儲存後清除 current_data 並刷新，無論成功或失敗都執行此步驟 
                    st.session_state.current_data = {
                        "inv_no": "N/A", "parsed_date": "N/A", "amount": 0, "pil_img": None
                    }
                    # 清除 hash，準備迎接下一個新檔案
                    st.session_state.last_uploaded_hash = None
                    st.rerun()
                    
    elif uploaded:
        st.info("點擊 '🧠 啟動 AI 辨識' 開始處理。")


# ========== TAB 2：分析與紀錄 (使用修正後的 load_data_for_dashboard) ==========
with tab2:
    st.header("📈 記帳分析儀表板")

    # 呼叫緩存函數，確保連線在函數內被管理和關閉
    df = load_data_for_dashboard()
    
    if len(df) == 0:
        st.info("尚無發票紀錄，請先到「掃描與記錄」分頁新增資料。")
        st.stop()

    # --- 1. 總覽 KPI ---
    total_spending = df['total_amount'].sum()
    st.subheader(f"總結 ({df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')})")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    col_kpi1.metric("總消費筆數", f"{len(df):,}")
    col_kpi2.metric("總消費金額", f"NT${total_spending:,.0f}")
    
    # 計算最近一個月的總支出
    # 處理 Period 類型比較
    if not df['YearMonth'].empty:
        latest_month_period = df['YearMonth'].max()
        df_latest_month = df[df['YearMonth'] == latest_month_period] 
        monthly_spending = df_latest_month['total_amount'].sum()
        col_kpi3.metric(f"{latest_month_period.strftime('%Y 年 %m 月')} 總開銷", f"NT${monthly_spending:,.0f}")
    
    st.markdown("---")


    # --- 2. 視覺化分析區 ---
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.subheader("💸 消費類別佔比")
        
        # 排除收入類別
        df_expense = df[df['category'] != '收入']
        category_summary = df_expense.groupby('category')['total_amount'].sum().reset_index()
        
        if len(category_summary) > 0:
            fig_pie = px.pie(
                category_summary,
                values='total_amount',
                names='category',
                title='各類別支出分佈',
                hole=.3, # 甜甜圈圖
                color_discrete_sequence=px.colors.qualitative.T10 
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, use_container_width=True) 
        else:
            st.info("暫無支出數據可供分析。")

    with col_chart2:
        st.subheader("📊 月度支出趨勢")
        
        # 按月度計算總和 (排除收入)
        monthly_trend = df_expense.groupby('YearMonth')['total_amount'].sum().reset_index()
        monthly_trend['Month'] = monthly_trend['YearMonth'].astype(str)
        
        if len(monthly_trend) > 0:
            fig_line = px.line(
                monthly_trend.sort_values(by='Month'), 
                x='Month',
                y='total_amount',
                title='月度支出總額趨勢',
                labels={'total_amount': '支出金額 (NT$)', 'Month': '月份'},
                markers=True,
                color_discrete_sequence=['#4c78a8']
            )
            fig_line.update_traces(line=dict(width=3))
            fig_line.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_line, use_container_width=True) 
        else:
            st.info("暫無歷史數據可供分析。")
            
    st.markdown("---")
    
    # --- 3. 歷史紀錄表格 (優化顯示) ---
    st.subheader("🧾 歷史帳目明細 (依月份整理)")
    
    pivot = df.sort_values(by=['date', 'created_at'], ascending=[False, False])
    
    display_cols = ['date', 'invoice_no', 'total_amount', 'category', 'note']
    display_name_map = {"date": "消費日期", "invoice_no": "發票號碼", "total_amount": "總金額 (NT$)", "category": "類別", "note": "項目/備註"}

    for period, group in pivot.groupby('YearMonth', sort=False):
        monthly_total = group['total_amount'].sum()
        
        with st.expander(f"📅 **{period.strftime('%Y 年 %m 月')}** — 總消費：NT${monthly_total:,.0f}", expanded=False):
            
            month_df = group[display_cols].rename(columns=display_name_map)
            month_df['消費日期'] = month_df['消費日期'].dt.strftime('%Y-%m-%d')
            
            month_df['總金額 (NT$)'] = month_df['總金額 (NT$)'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(
                month_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "總金額 (NT$)": st.column_config.TextColumn(
                        "總金額 (NT$)",
                        help="本筆消費金額",
                        disabled=True
                    )
                }
            )