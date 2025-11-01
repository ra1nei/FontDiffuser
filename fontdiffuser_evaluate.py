import os
import json
import zipfile
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import lpips
import torch_fidelity

from sample import sampling, load_fontdiffuer_pipeline
from utils import save_image_with_content_style

# LPIPS model (khởi tạo global)
lpips_model = lpips.LPIPS(net='alex')

# ======================
# UTILS
# ======================
def preprocess_image(path, size, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)[None, :].to(device)

def collect_images(root_dir):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.exists(root_dir) else []

def load_image_tensor(path, size=(96, 96), device="cuda"):
    img = Image.open(path).convert("RGB").resize(size)
    return img, transforms.ToTensor()(img).unsqueeze(0).to(device)

def save_single_image(save_dir, image, filename):
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, filename))

def compute_metrics(gen_pil, target_pil, gen_t, target_t):
    img1, img2 = np.array(gen_pil), np.array(target_pil)
    try:
        ssim_val = ssim(img1, img2, channel_axis=-1, data_range=255)
    except Exception:
        ssim_val = float('nan')
    lpips_val = lpips_model(gen_t, target_t).item()
    l1_val = F.l1_loss(gen_t, target_t).item()
    return {"SSIM": ssim_val, "LPIPS": lpips_val, "L1": l1_val}

# ======================
# MAIN SAMPLING
# ======================
def batch_sampling(args):
    pipe = load_fontdiffuer_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(123)

    chinese_images = collect_images(args.chinese_dir)
    print(f"Tổng số ảnh Chinese: {len(chinese_images)}")

    samples = []
    for chi_path in chinese_images:
        font_name = os.path.basename(os.path.dirname(chi_path))  # tên font
        glyph_name = os.path.basename(chi_path)  # ví dụ: 汉.png

        # content (source glyph)
        content_path = os.path.join(args.source_dir, glyph_name)
        if not os.path.exists(content_path):
            continue

        # style (english A.png cùng font)
        style_path = os.path.join(args.english_dir, font_name, "A.png")
        if not os.path.exists(style_path):
            continue

        samples.append({
            "content": content_path,
            "style": style_path,
            "target": chi_path
        })

    print(f"Tổng số mẫu hợp lệ: {len(samples)}")

    gen_dir = os.path.join(args.save_dir, "generated")
    real_dir = os.path.join(args.save_dir, "target")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    metrics_list = []

    for i, s in enumerate(tqdm(samples, desc="Sampling")):
        content_path, style_path, target_path = s["content"], s["style"], s["target"]

        content_img = preprocess_image(content_path, args.content_image_size, args.device)
        style_img = preprocess_image(style_path, args.style_image_size, args.device)

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
                dm_size=args.content_image_size[0],
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

        out_name = f"{i:04d}_gen.jpg"
        save_single_image(gen_dir, out_pil, out_name)

        save_image_with_content_style(
            save_dir=args.save_dir,
            image=out_pil,
            content_image_pil=None,
            content_image_path=content_path,
            style_image_path=style_path,
            resolution=args.content_image_size[0],
            filename=f"{i:04d}_compare.jpg"
        )

        target_pil, target_t = load_image_tensor(target_path, args.content_image_size, args.device)
        gen_t = transforms.ToTensor()(out_pil).unsqueeze(0).to(args.device)
        metrics = compute_metrics(out_pil, target_pil, gen_t, target_t)
        metrics.update({
            "sample_id": i,
            "output": os.path.join(gen_dir, out_name),
            "target": target_path,
            "content": content_path,
            "style": style_path
        })
        metrics_list.append(metrics)
        target_pil.save(os.path.join(real_dir, os.path.basename(target_path)))

    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)

    # Mean metrics
    mean_vals = {k: np.nanmean([m[k] for m in metrics_list if k in m]) for k in ["SSIM", "LPIPS", "L1"]}
    print("\nEvaluation Summary:")
    for k, v in mean_vals.items():
        print(f"{k}: {v:.4f}")

    # FID
    metrics_fid = torch_fidelity.calculate_metrics(
        input1=gen_dir, input2=real_dir,
        cuda=True, isc=False, fid=True,
        batch_size=16, save_cpu_ram=True, num_workers=0
    )
    fid_value = metrics_fid.get("frechet_inception_distance", float("nan"))
    print(f"\nFID: {fid_value:.4f}")

    # Save summary
    with open(os.path.join(args.save_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Metric\tValue\n")
        f.write(f"SSIM\t{mean_vals['SSIM']:.6f}\n")
        f.write(f"LPIPS\t{mean_vals['LPIPS']:.6f}\n")
        f.write(f"L1\t{mean_vals['L1']:.6f}\n")
        f.write(f"FID\t{fid_value:.6f}\n")

    # ZIP
    zip_path = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.save_dir):
            for f in files:
                if f.endswith((".jpg", ".png", ".json", ".txt")):
                    zf.write(os.path.join(root, f), arcname=os.path.relpath(os.path.join(root, f), args.save_dir))
    print(f"Results saved to {zip_path}")

# ======================
# ENTRY
# ======================
def main():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--english_dir", type=str, required=True)
    parser.add_argument("--chinese_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_batch", action="store_true")
    # parser.add_argument("--lexicon_txt", type=str, required=True)
    # parser.add_argument("--start_chinese_idx", type=int, required=True)
    # parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, f"{args.name}_all_{datetime.now():%H-%M-%S_%d-%m}")
    os.makedirs(args.save_dir, exist_ok=True)
    args.style_image_size = args.content_image_size = (96, 96)
    lpips_model.to(args.device)

    batch_sampling(args)

if __name__ == "__main__":
    main()