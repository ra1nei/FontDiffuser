import os
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

from accelerate.utils import set_seed
import lpips
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# import pipeline + model utils
from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)


def preprocess_image(img_path, size):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img)[None, :]


def load_pipeline(args):
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    model.to(args.device)

    train_scheduler = build_ddpm_scheduler(args=args)
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    return pipe


def evaluate_pair(fake_img, real_img, device="cuda"):
    fake = (fake_img + 1) / 2 if fake_img.min() < 0 else fake_img
    real = (real_img + 1) / 2 if real_img.min() < 0 else real_img

    if fake.shape[-2:] != real.shape[-2:]:
        _, _, H, W = real.shape
        fake = torch.nn.functional.interpolate(fake, size=(H, W), mode="bilinear", align_corners=False)

    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_value = ssim_fn(fake, real).item()

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_value = lpips_fn((fake * 2 - 1), (real * 2 - 1)).item()

    l1_value = F.l1_loss(fake, real).item()
    return ssim_value, lpips_value, l1_value


def evaluate_ufsc(args):
    pipe = load_pipeline(args)

    ssim_scores, lpips_scores, l1_scores = [], [], []

    # Lấy toàn bộ style ảnh từ chinese + english
    font_dirs = [os.path.join(args.english_dir, f) for f in os.listdir(args.english_dir)] + \
                [os.path.join(args.chinese_dir, f) for f in os.listdir(args.chinese_dir)]

    style_images = []
    for font_dir in font_dirs:
        for f in os.listdir(font_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                style_images.append(os.path.join(font_dir, f))

    source_files = [os.path.join(args.source_dir, f) for f in os.listdir(args.source_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for style_path in tqdm(style_images, desc="Evaluating UF-SC"):
        if not source_files:
            continue
        content_path = random.choice(source_files)

        # preprocess
        content_img = preprocess_image(content_path, args.content_image_size).to(args.device)
        style_img = preprocess_image(style_path, args.style_image_size).to(args.device)
        gt_img = preprocess_image(style_path, args.content_image_size).to(args.device)  # groundtruth = style ảnh

        with torch.no_grad():
            fake_imgs = pipe.generate(
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

        ssim, lp, l1 = evaluate_pair(fake_imgs[0].unsqueeze(0), gt_img, args.device)
        ssim_scores.append(ssim)
        lpips_scores.append(lp)
        l1_scores.append(l1)

    print(f"\n==== UF-SC Evaluation Results ====")
    if ssim_scores:
        print(f"SSIM : {np.mean(ssim_scores):.4f}")
        print(f"LPIPS: {np.mean(lpips_scores):.4f}")
        print(f"L1   : {np.mean(l1_scores):.4f}")
    else:
        print("⚠️ Không có mẫu nào để evaluate!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--english_dir", type=str, required=True)
    parser.add_argument("--chinese_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    set_seed(args.seed)

    evaluate_ufsc(args)


if __name__ == "__main__":
    main()