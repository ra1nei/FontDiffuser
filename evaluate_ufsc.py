import os
import random
import torch
import numpy as np
import json
import zipfile

import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from sample import sampling
from accelerate.utils import set_seed
from utils import save_image_with_content_style
from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)



def preprocess_image(img_path, size):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img)[None, :]



def load_pipeline(args):
    unet = build_unet(args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location=args.device))

    style_encoder = build_style_encoder(args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location=args.device))

    content_encoder = build_content_encoder(args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location=args.device))

    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
    ).to(args.device)

    scheduler = build_ddpm_scheduler(args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    return pipe



def get_style_images(english_dir, chinese_dir):
    font_dirs = []
    for base in [english_dir, chinese_dir]:
        for f in os.listdir(base):
            full_path = os.path.join(base, f)
            if os.path.isdir(full_path):
                font_dirs.append(full_path)

    style_images = []
    for font_dir in font_dirs:
        for fname in os.listdir(font_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                style_images.append(os.path.join(font_dir, fname))
    return style_images



def batch_sampling(args):
    pipe = load_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    set_seed(args.seed)

    # Load dataset
    english_contents = [
        os.path.join(args.english_dir, fnt, fname)
        for fnt in os.listdir(args.english_dir)
        for fname in os.listdir(os.path.join(args.english_dir, fnt))
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    chinese_contents = [
        os.path.join(args.chinese_dir, fnt, fname)
        for fnt in os.listdir(args.chinese_dir)
        for fname in os.listdir(os.path.join(args.chinese_dir, fnt))
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    english_styles = chinese_contents
    chinese_styles = english_contents
    print(f"🖋 English content: {len(english_contents)} | Chinese content: {len(chinese_contents)}")

    # JSON path
    json_path = os.path.join(args.save_dir, "samples.json")

    # Load or create new sample list
    if os.path.exists(json_path):
        print(f"Reusing existing batch from: {json_path}")
        with open(json_path, "r") as f:
            samples = json.load(f)
    else:
        print(f"Creating new random batch ({args.num_samples} samples)")
        all_contents = english_contents + chinese_contents
        samples = []
        for _ in range(args.num_samples):
            content_path = random.choice(all_contents)
            is_english = "english" in content_path.lower()
            style_path = random.choice(chinese_styles) if is_english else random.choice(english_styles)
            samples.append({"content": content_path, "style": style_path})
        with open(json_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"Saved batch to {json_path}")

    # Sampling loop
    for i, sample in enumerate(tqdm(samples, desc="Sampling")):
        content_path = sample["content"]
        style_path = sample["style"]

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
        out_pil = Image.fromarray(((out_img / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

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

        os.rename(os.path.join(args.save_dir, "out_with_cs.jpg"), os.path.join(args.save_dir, out_name))

    zip_path = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}.zip")
    print(f"\nCompressing results to {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.save_dir):
            for file in files:
                zipf.write(os.path.join(root, file),
                           arcname=os.path.relpath(os.path.join(root, file), args.save_dir))
    print(f"Results saved and zipped at: {zip_path}")



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
    args = parser.parse_args()

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    batch_sampling(args)


if __name__ == "__main__":
    main()