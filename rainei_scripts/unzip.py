"""
Dùng để unzip
"""

import os
import zipfile

def main():
    zip_filename = "thesis-data-jpg.zip"
    extract_dir = "thesis-data-jpg"

    zip_path = os.path.join(os.path.dirname(__file__), zip_filename)

    if not os.path.exists(zip_path):
        print(f"File zip không tồn tại: {zip_path}")
        return

    print(f"Đang giải nén '{zip_filename}' vào thư mục '{extract_dir}'...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅")
    except zipfile.BadZipFile:
        print("❌ Lỗi: File không phải là một tệp .zip hợp lệ.")
        return

if __name__ == "__main__":
    main()
