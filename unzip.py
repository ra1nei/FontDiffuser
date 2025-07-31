# zip_folder.py
import zipfile
import os

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, start=folder_path)
                zipf.write(filepath, arcname)

# Example usage
zip_folder('thesis-data', 'thesis-data.zip')
