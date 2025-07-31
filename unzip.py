import os
import zipfile
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

    # Giải nén
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"✅ Unzipped successfully into: {extract_dir}")
    except zipfile.BadZipFile:
        print("❌ Lỗi: File không phải là một tệp .zip hợp lệ.")

if __name__ == "__main__":
    main()

    with zipfile.ZipFile("thesis-data.zip", 'r') as zip_ref:
    for name in zip_ref.namelist():
        print(name)