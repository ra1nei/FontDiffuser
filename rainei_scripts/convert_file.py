"""
Dùng để convert data của bài báo Font Style Transfer sang cấu trúc mong muốn
"""

import os
import shutil
from tqdm import tqdm

INPUT_ROOT = r"D:\style-transfer-font\train"
OUTPUT_ROOT = r"D:\converted-train"
CONTENT_DIR = os.path.join(OUTPUT_ROOT, "ContentImage")
TARGET_DIR = os.path.join(OUTPUT_ROOT, "TargetImage")

os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(TARGET_DIR, exist_ok=True)

copied_content = set()

for lang in ["chinese", "english"]:
    lang_folder = os.path.join(INPUT_ROOT, lang)
    for fontname in tqdm(os.listdir(lang_folder), desc=f"Processing {lang}"):
        font_path = os.path.join(lang_folder, fontname)
        if not os.path.isdir(font_path):
            continue

        style_name = f"{fontname}_{lang}"
        output_style_dir = os.path.join(TARGET_DIR, style_name)
        os.makedirs(output_style_dir, exist_ok=True)

        for fname in os.listdir(font_path):
            if not fname.lower().endswith(".png"):
                continue
            char_name = os.path.splitext(fname)[0]
            full_input_path = os.path.join(font_path, fname)

            target_name = f"{style_name}+{char_name}.jpg"
            shutil.copyfile(full_input_path, os.path.join(output_style_dir, target_name))

            if char_name not in copied_content:
                shutil.copyfile(full_input_path, os.path.join(CONTENT_DIR, f"{char_name}.jpg"))
                copied_content.add(char_name)

print("Finish!")
