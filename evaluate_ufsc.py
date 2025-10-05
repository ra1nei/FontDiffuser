import os
import random
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)
from utils import save_single_image


# -----------------------------
# Tiền xử lý ảnh
# -----------------------------
def preprocess_image(img_path, size):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img)[None, :]


# -----------------------------
# Load pipeline
# -----------------------------
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


# -----------------------------
# Lấy danh sách ảnh style
# -----------------------------
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


# -----------------------------
# Sampling cho toàn bộ dataset
# -----------------------------
def batch_sampling(args):
    pipe = load_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # load content và style
    content_images = [
        os.path.join(args.source_dir, f)
        for f in os.listdir(args.source_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    style_images = get_style_images(args.english_dir, args.chinese_dir)

    print(f"🖋 Found {len(content_images)} content images")
    print(f"🎨 Found {len(style_images)} style images")

    # Loop qua mỗi style font
    for i, style_path in enumerate(tqdm(style_images, desc="Sampling")):
        content_path = random.choice(content_images)

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

        # Lưu ảnh ra
        out_img = out_imgs[0]
        font_name = os.path.basename(os.path.dirname(style_path))
        out_name = f"{font_name}_{i:04d}.png"
        save_path = os.path.join(args.save_dir, out_name)
        save_single_image(args.save_dir, out_img)
    print(f"\n✅ Done! Generated images saved in: {args.save_dir}")


# -----------------------------
# Main
# -----------------------------
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    batch_sampling(args)


if __name__ == "__main__":
    main()
