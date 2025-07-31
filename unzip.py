import zipfile
import os

zip_path = "thesis-data.zip"
extract_dir = "thesis-data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("✅ Unzipped successfully into:", extract_dir)
