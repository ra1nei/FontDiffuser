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

# LPIPS model (đặt lên device sau khi parse args nếu cần nhiều GPU)
lpips_model = lpips.LPIPS(net='alex').to("cuda")


def preprocess_image(path, size, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)[None, :].to(device)


def collect_images(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(root, fname))
    return paths



def get_lang_from_path(path, english_dir, chinese_dir):
    """Cố gắng xác định 'english'/'chinese' từ path so sánh với english_dir/chinese_dir."""
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
    """
    Tìm target path tương ứng:
    - style_font là tên folder của style_path (parent folder của style image)
    - ưu tiên tìm target under english_dir/style_font/<char_filename> nếu tồn tại,
      nếu không, thử chinese_dir/style_font/<char_filename>.
    - Nếu cả hai không có, trả về đường dẫn 'expected' (english_dir/style_font/filename).
    """
    style_font = os.path.basename(os.path.dirname(style_path))
    char_filename = os.path.basename(content_path)
    eng_candidate = os.path.join(english_dir, style_font, char_filename)
    chi_candidate = os.path.join(chinese_dir, style_font, char_filename)

    if os.path.exists(eng_candidate):
        return eng_candidate
    if os.path.exists(chi_candidate):
        return chi_candidate
    # fallback: nếu không thấy, trả về eng_candidate làm đường dẫn mong đợi
    return eng_candidate


def compute_metrics(gen_pil, target_pil, gen_t, target_t):
    img1 = np.array(gen_pil)
    img2 = np.array(target_pil)
    try:
        ssim_val = ssim(img1, img2, channel_axis=-1, data_range=255)
    except Exception:
        # fallback nếu hình có format khác
        ssim_val = float(np.nan)
    lpips_val = lpips_model(gen_t, target_t).item()
    l1_val = F.l1_loss(gen_t, target_t).item()
    return {"SSIM": float(ssim_val), "LPIPS": float(lpips_val), "L1": float(l1_val)}


def load_image_tensor(path, size=(96, 96), device="cuda"):
    img = Image.open(path).convert("RGB").resize(size)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
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

    # --- source content là từ source_dir ---
    source_contents = collect_images(args.source_dir)
    # dùng english/chinese dirs để tìm target; styles lấy từ english/chinese dirs too (bằng cách collect images there)
    english_contents = collect_images(args.english_dir)
    chinese_contents = collect_images(args.chinese_dir)
    # styles pools: lấy tất cả image paths (các ảnh style nằm trong thư mục của style)
    english_style_images = collect_images(args.english_dir)
    chinese_style_images = collect_images(args.chinese_dir)

    # NOTE: trước code có english_styles = chinese_contents và ngược lại — giữ logic nếu bạn muốn cross-lang styles.
    # Ở đây để đơn giản, ta dùng style pool tương ứng với language (có thể sửa sao cho cross nếu cần).
    english_styles = english_style_images
    chinese_styles = chinese_style_images

    print(f"Source contents: {len(source_contents)} | English content (target dir sample): {len(english_contents)} | Chinese content (target dir sample): {len(chinese_contents)}")

    json_path = os.path.join(args.save_dir, "samples.json")

    # Get the JSON file for later inference using different models
    if os.path.exists(json_path) and args.use_batch:
        print(f"Reusing existing batch from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print(f"Creating new batch with {args.num_samples} samples (content from source_dir)")
        samples = []
        num_per_lang = args.num_samples // 2
        max_retry = 1000

        def get_valid_pair_from_source(content_pool, style_pool, content_lang_name):
            for _ in range(max_retry):
                content = random.choice(content_pool)
                style = random.choice(style_pool)
                # lọc content cùng lang
                if get_lang_from_path(content, args.english_dir, args.chinese_dir) != content_lang_name:
                    continue
                target_path = get_target_path(content, style, args.english_dir, args.chinese_dir)
                if os.path.exists(target_path):
                    return {"content": content, "style": style, "target": target_path}


        # tạo samples: nửa từ "english" (dựa trên target dir) và nửa từ "chinese"
        # Lưu ý: content được chọn từ source_contents, nên phải có đủ variety trong source_dir
        for _ in range(num_per_lang):
            samples.append(get_valid_pair_from_source(source_contents, english_styles, "english"))
        for _ in range(num_per_lang):
            samples.append(get_valid_pair_from_source(source_contents, chinese_styles, "chinese"))
        # nếu lẻ, thêm 1 sample nữa
        while len(samples) < args.num_samples:
            samples.append(get_valid_pair_from_source(source_contents, english_styles, "english"))

        # Save samples with full paths (content from source_dir)
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
        content_path, style_path, target_path = s["content"], s["style"], s.get("target")
        content_img = preprocess_image(content_path, args.content_image_size, device=args.device)
        style_img = preprocess_image(style_path, args.style_image_size, device=args.device)

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

        # Save compare image: content(from source), style, generated
        save_image_with_content_style(
            save_dir=args.save_dir,
            image=out_pil,
            content_image_pil=None,
            content_image_path=content_path,
            style_image_path=style_path,
            resolution=args.content_image_size[0],
            filename=f"{i:04d}_compare.jpg"
        )

        # target_path was precomputed when creating samples; but recompute if missing key
        if target_path is None:
            target_path = get_target_path(content_path, style_path, args.english_dir, args.chinese_dir)

        # DEBUG
        # print(f"Target path: {target_path}")
        if os.path.exists(target_path):
            target_pil, target_t = load_image_tensor(target_path, args.content_image_size, device=args.device)
            gen_t = transforms.ToTensor()(out_pil).unsqueeze(0).to(args.device)
            metrics = compute_metrics(out_pil, target_pil, gen_t, target_t)
            metrics["sample_id"] = i
            metrics["output"] = out_path
            metrics["target"] = target_path
            metrics["content"] = content_path
            metrics["style"] = style_path
            metrics_list.append(metrics)
            target_pil.save(os.path.join(real_dir, os.path.basename(target_path)))
        else:
            print(f"⚠️ Missing target for sample {i}: {target_path}")

    # Save per-sample metrics JSON
    metrics_path = os.path.join(args.save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_path}")

    # Compute mean SSIM/LPIPS/L1
    mean_vals = {}
    if metrics_list:
        mean_vals = {k: np.nanmean([m[k] for m in metrics_list if k in m]) for k in ["SSIM", "LPIPS", "L1"]}
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

    fid_value = metrics_fid.get("frechet_inception_distance", float("nan"))
    print(f"\nFID: {fid_value:.4f}")

    # Save summary TXT containing 4 metrics
    summary_txt = os.path.join(args.save_dir, "metrics_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Metric\tValue\n")
        f.write(f"SSIM\t{mean_vals.get('SSIM', float('nan')):.6f}\n")
        f.write(f"LPIPS\t{mean_vals.get('LPIPS', float('nan')):.6f}\n")
        f.write(f"L1\t{mean_vals.get('L1', float('nan')):.6f}\n")
        f.write(f"FID\t{fid_value:.6f}\n")
    print(f"Saved metrics summary to {summary_txt}")

    # ZIP (lưu cả json + ảnh + txt)
    zip_path = os.path.join(os.path.dirname(args.save_dir), f"{os.path.basename(args.save_dir)}.zip")
    print(f"Compressing results to {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(args.save_dir):
            for f in files:
                if f.endswith((".jpg", ".png", ".json", ".txt")):
                    zf.write(os.path.join(root, f), arcname=os.path.relpath(os.path.join(root, f), args.save_dir))
    print(f"Results saved to {zip_path}")


def main():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Folder chứa content source (thay vì lấy trực tiếp từ content folders).")
    parser.add_argument("--english_dir", type=str, required=True)
    parser.add_argument("--chinese_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--use_batch", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, f"result_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)

    args.style_image_size = (96, 96)
    args.content_image_size = (96, 96)

    # set device for LPIPS and other tensors
    args.device = args.device if args.device is not None else "cuda"
    lpips_model.to(args.device)

    batch_sampling(args)


if __name__ == "__main__":
    main()