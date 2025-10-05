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
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import lpips
import torch_fidelity
from datetime import datetime

from sample import sampling, load_fontdiffuer_pipeline
from utils import save_args_to_yaml, save_image_with_content_style

lpips_model = lpips.LPIPS(net='alex').to("cuda")


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

    parts = os.path.normpath(p).lower().split(os.sep)
    if "english" in parts:
        return "english"
    if "chinese" in parts:
        return "chinese"
    return None


def get_target_path(content_path, style_path, english_dir, chinese_dir):
    content_lang = get_lang_from_path(content_path, english_dir, chinese_dir)
    style_font = os.path.basename(os.path.dirname(style_path))
    char_filename = os.path.basename(content_path)
    base_dir = chinese_dir if content_lang == "chinese" else english_dir
    return os.path.join(base_dir, style_font, char_filename)


def compute_metrics(gen_pil, target_pil, gen_t, target_t):
    img1 = np.array(gen_pil)
    img2 = np.array(target_pil)
    ssim_val = ssim(img1, img2, channel_axis=-1, data_range=255)
    lpips_val = lpips_model(gen_t, target_t).item()
    l1_val = F.l1_loss(gen_t, target_t).item()
    return {"SSIM": float(ssim_val), "LPIPS": float(lpips_val), "L1": float(l1_val)}


def load_image_tensor(path, size=(96, 96)):
    img = Image.open(path).convert("RGB").resize(size)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to("cuda")
    return img, tensor


def save_single_image(save_dir, image, filename="out_single.png"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)
    return save_path


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_path, resolution, filename="out_with_cs.jpg"):
    os.makedirs(save_dir, exist_ok=True)
    new_image = Image.new('RGB', (resolution * 3, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution * 2, 0))
    save_path = os.path.join(save_dir, filename)
    new_image.save(save_path)
    return save_path


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

    # Get the JSON file for later inference using different models
    if os.path.exists(json_path) and args.use_batch:
        print(f"Reusing existing batch from {json_path}")
        with open(json_path, "r") as f:
            samples = json.load(f)
    else:
        print(f"Creating new batch with {args.num_samples} samples")
        samples = []
        num_per_lang = args.num_samples // 2
        max_retry = 1000

        def get_valid_pair(content_pool, style_pool, content_lang):
            for _ in range(max_retry):
                content = random.choice(content_pool)
                style = random.choice(style_pool)
                target_path = get_target_path(content, style, args.english_dir, args.chinese_dir)
                if os.path.exists(target_path):
                    return {"content": content, "style": style}
            raise RuntimeError(f"Không tìm được cặp hợp lệ cho {content_lang} sau {max_retry} lần thử!")

        for _ in range(num_per_lang):
            samples.append(get_valid_pair(english_contents, english_styles, "english"))
        for _ in range(num_per_lang):
            samples.append(get_valid_pair(chinese_contents, chinese_styles, "chinese"))
        if len(samples) < args.num_samples:
            samples.append(get_valid_pair(english_contents, english_styles, "english"))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"Saved sample batch to {json_path}")

    # Main Inference
    gen_dir = os.path.join(args.save_dir, "generated")
    real_dir = os.path.join(args.save_dir, "target")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)

    metrics_list = []

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

        out_name = f"{i:04d}_gen.jpg"
        out_path = os.path.join(gen_dir, out_name)
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

        target_path = get_target_path(content_path, style_path, args.english_dir, args.chinese_dir)
        # DEBUG
        print(target_path)
        if os.path.exists(target_path):
            target_pil, target_t = load_image_tensor(target_path, args.content_image_size)
            gen_t = transforms.ToTensor()(out_pil).unsqueeze(0).to("cuda")
            metrics = compute_metrics(out_pil, target_pil, gen_t, target_t)
            metrics["sample_id"] = i
            metrics["output"] = out_path
            metrics["target"] = target_path
            metrics_list.append(metrics)
            target_pil.save(os.path.join(real_dir, os.path.basename(target_path)))
        else:
            print(f"⚠️ Missing target for sample {i}: {target_path}")

    # Calculate metrics
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_path}")

    if metrics_list:
        mean_vals = {k: np.mean([m[k] for m in metrics_list if k in m]) for k in ["SSIM", "LPIPS", "L1"]}
        print("\nEvaluation Summary:")
        for k, v in mean_vals.items():
            print(f"{k}: {v:.4f}")

    # Calculating FID
    def find_non_images(folder):
        bad = []
        for f in os.listdir(folder):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                bad.append(f)
        return bad

    for folder in [gen_dir, real_dir]:
        bad_files = find_non_images(folder)
        for bf in bad_files:
            os.remove(os.path.join(folder, bf))

    metrics_fid = torch_fidelity.calculate_metrics(
        input1=gen_dir,
        input2=real_dir,
        cuda=True,
        isc=False,
        fid=True,
        batch_size=16,
        save_cpu_ram=True,
        num_workers=0
    )

    fid_value = metrics_fid["frechet_inception_distance"]
    print(f"\nFID: {fid_value:.4f}")

    # ZIP
    zip_path = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}.zip")
    print(f"Compressing results to {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.save_dir):
            for f in files:
                if f.endswith((".jpg", ".png", ".json")):
                    zf.write(os.path.join(root, f), arcname=os.path.relpath(os.path.join(root, f), args.save_dir))
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
    parser.add_argument("--use_batch", action="store_true")
    args = parser.parse_args()

    now = datetime.now().strftime("%H%M%S_%d%m%Y")
    args.save_dir = args.save_dir or f"results_{now}"

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    batch_sampling(args)


if __name__ == "__main__":
    main()
