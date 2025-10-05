import os
import random
import torch
import json
import zipfile
import numpy as np
from tqdm import tqdm
from PIL import Image
from accelerate.utils import set_seed
from torchvision import transforms

from sample import sampling, load_fontdiffuer_pipeline
from utils import save_args_to_yaml, save_image_with_content_style, save_single_image


def preprocess_image(path, size):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)[None, :]


def collect_images(root_dir):
    paths = []
    for folder in os.listdir(root_dir):
        full_dir = os.path.join(root_dir, folder)
        if os.path.isdir(full_dir):
            for fname in os.listdir(full_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    paths.append(os.path.join(full_dir, fname))
    return paths



def get_lang_from_path(path, english_dir, chinese_dir):
    p = os.path.abspath(path)
    eng = os.path.abspath(english_dir)
    chi = os.path.abspath(chinese_dir)

    # try commonpath (most robust)
    try:
        if os.path.commonpath([p, eng]) == eng:
            return "english"
    except ValueError:
        pass
    try:
        if os.path.commonpath([p, chi]) == chi:
            return "chinese"
    except ValueError:
        pass

    # fallback: check path components for exact folder name
    parts = os.path.normpath(p).lower().split(os.sep)
    if "english" in parts:
        return "english"
    if "chinese" in parts:
        return "chinese"
    return None



def batch_sampling(args):
    pipe = load_fontdiffuer_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    set_seed(args.seed)

    english_contents = collect_images(args.english_dir)
    chinese_contents = collect_images(args.chinese_dir)
    english_styles = chinese_contents
    chinese_styles = english_contents

    print(f"English content: {len(english_contents)} | Chinese content: {len(chinese_contents)}")

    json_path = os.path.join(args.save_dir, "samples.json")

    if os.path.exists(json_path) and args.use_batch == True:
        print(f"Reusing existing batch from {json_path}")
        with open(json_path, "r") as f:
            samples = json.load(f)
    else:
        print(f"Creating new batch with {args.num_samples} samples")
        all_contents = english_contents + chinese_contents
        samples = []
        for _ in range(args.num_samples):
            content = random.choice(all_contents)
            lang = get_lang_from_path(content, args.english_dir, args.chinese_dir)
            if lang == "english":
                style = random.choice(chinese_contents)
            elif lang == "chinese":
                style = random.choice(english_contents)
            samples.append({"content": content, "style": style})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Saved sample batch to {json_path}")

    # DEBUG
    for s in samples[:10]:
        print("Content:", s["content"])
        print(" -> lang:", get_lang_from_path(s["content"], args.english_dir, args.chinese_dir))
        print("Style :", s["style"])


    for i, s in enumerate(tqdm(samples, desc="Sampling")):
        content_path, style_path = s["content"], s["style"]
        content_img = preprocess_image(content_path, args.content_image_size).to(args.device)
        style_img = preprocess_image(style_path, args.style_image_size).to(args.device)

        with torch.no_grad():
            out_imgs = pipe.generate(
                content_images=content_img,
                style_images=style_img,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size,
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn,
            )

        out_img = out_imgs[0]
        if isinstance(out_img, torch.Tensor):
            out_pil = Image.fromarray(
                ((out_img / 2 + 0.5).clamp(0, 1)
                 .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            )
        else:
            out_pil = out_img

        content_font = os.path.basename(os.path.dirname(content_path))
        style_font = os.path.basename(os.path.dirname(style_path))
        out_name = f"{content_font}_to_{style_font}_{i:04d}.jpg"

        save_image_with_content_style(
            save_dir=args.save_dir,
            image=out_pil,
            content_image_pil=None,
            content_image_path=content_path,
            style_image_path=style_path,
            resolution=args.content_image_size[0],
        )

        save_single_image(
            save_dir=args.save_dir, image=out_imgs[0],
        )

        os.rename(os.path.join(args.save_dir, "out_with_cs.jpg"), os.path.join(args.save_dir, out_name))

    zip_path = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}.zip")
    print(f"Compressing results to {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.save_dir):
            for f in files:
                zf.write(os.path.join(root, f),
                         arcname=os.path.relpath(os.path.join(root, f), args.save_dir))
    print(f"Results saved to {zip_path}")


def main():
    import argparse
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--english_dir", type=str, required=True)
    parser.add_argument("--chinese_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--use_batch", action="store_true", help="Reuse existing batch if available")
    args = parser.parse_args()

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    batch_sampling(args)


if __name__ == "__main__":
    main()
