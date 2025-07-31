import os
import zipfile

def main():
    zip_filename = "thesis-data.zip"
    extract_dir = "thesis-data"

    # Lấy đường dẫn tuyệt đối đến file zip
    zip_path = os.path.join(os.path.dirname(__file__), zip_filename)

    # Kiểm tra file zip có tồn tại không
    if not os.path.exists(zip_path):
        print(f"❌ File zip không tồn tại: {zip_path}")
        return

    # Bắt đầu giải nén
    print(f"🔄 Đang giải nén '{zip_filename}' vào thư mục '{extract_dir}'...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Đã giải nén thành công vào: {extract_dir}")
    except zipfile.BadZipFile:
        print("❌ Lỗi: File không phải là một tệp .zip hợp lệ.")
        return

    # In danh sách file đã giải nén
    print("\n📂 Danh sách các file trong zip:")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            print(f" - {name}")

if __name__ == "__main__":
    main()
