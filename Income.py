import csv
import os
from datetime import datetime

# 定義檔案名稱和欄位名稱 (標題列)
FILE_NAME = 'income_records.csv'
FIELDNAMES = ['Date', 'Amount', 'Source', 'Category']

def load_income():
    """
    從 CSV 檔案中載入所有收入記錄。
    如果檔案不存在，則建立一個新的空檔案並寫入標題。
    """
    records = []
    if not os.path.exists(FILE_NAME):
        print(f"⚠️ 檔案 {FILE_NAME} 不存在，已建立新的檔案。")
        with open(FILE_NAME, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return records

    try:
        with open(FILE_NAME, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 確保金額欄位是浮點數，方便後續計算
                row['Amount'] = float(row['Amount'])
                records.append(row)
    except Exception as e:
        print(f"❌ 讀取檔案時發生錯誤: {e}")
    return records

def save_income(new_record):
    """
    將一筆新的收入記錄寫入 CSV 檔案。
    """
    try:
        # 使用 'a' 模式 (append) 在檔案末尾追加新行
        with open(FILE_NAME, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            # 確保金額格式正確，方便儲存
            new_record['Amount'] = f"{new_record['Amount']:.2f}"
            writer.writerow(new_record)
        print(f"✅ 成功新增並儲存收入：{new_record['Source']} - ${new_record['Amount']}")
    except Exception as e:
        print(f"❌ 寫入檔案時發生錯誤: {e}")

def add_new_income():
    """
    提示使用者輸入收入資訊並儲存。
    """
    print("\n--- 新增收入記錄 ---")
    
    # 獲取日期，如果使用者不輸入，則使用今天日期
    date_str = input(f"日期 (YYYY-MM-DD, 留空則為今天 {datetime.now().strftime('%Y-%m-%d')}): ")
    if not date_str:
        date_str = datetime.now().strftime('%Y-%m-%d')
        
    while True:
        try:
            amount_str = input("金額 (請輸入數字): ")
            amount = float(amount_str)
            if amount <= 0:
                 raise ValueError
            break
        except ValueError:
            print("🚫 金額輸入無效，請輸入一個大於零的數字。")

    source = input("描述/來源 (例如: 11月薪水): ")
    category = input("分類 (例如: 薪資, 投資, 兼職, 贈與): ")

    new_record = {
        'Date': date_str,
        'Amount': amount,
        'Source': source,
        'Category': category
    }
    save_income(new_record)

def show_all_income(records):
    """
    顯示所有收入記錄並計算總計。
    """
    if not records:
        print("\n目前沒有任何收入記錄。")
        return

    print("\n--- 📝 所有收入記錄 ---")
    total_income = 0
    
    # 格式化輸出標題
    header = f"{'日期':<12} | {'金額':<10} | {'來源/描述':<20} | {'分類':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for record in records:
        # 由於讀取時已轉為 float，這裡可以直接計算
        amount = record['Amount']
        total_income += amount
        
        # 格式化輸出每筆記錄
        print(f"{record['Date']:<12} | ${amount:<9.2f} | {record['Source']:<20} | {record['Category']:<10}")

    print("-" * len(header))
    print(f"✨ 總收入合計：${total_income:.2f}")


# --- 主程式運行區塊 ---
if __name__ == "__main__":
    
    # 1. 載入現有資料
    current_records = load_income()
    
    # 2. 顯示現有收入
    show_all_income(current_records)
    
    # 3. 詢問是否新增
    if input("\n是否要新增一筆收入？ (y/n): ").lower() == 'y':
        add_new_income()
        
        # 4. 新增後，再次載入並顯示新的總計
        updated_records = load_income()
        show_all_income(updated_records)
        
    print("\n程式運行結束。")