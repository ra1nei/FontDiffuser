import os
import random
import torch
from PIL import Image
from datetime import datetime
from torchvision import transforms

from sample import load_fontdiffuer_pipeline


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
    """Thu thập toàn bộ ảnh .png, .jpg, .jpeg trong thư mục"""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.exists(root_dir) else []


def save_single_image(save_dir, image, filename):
    """Lưu ảnh PIL vào thư mục"""
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, filename))


def load_image_tensor(path, size=(96, 96)):
    """Đọc ảnh groundtruth để resize và lưu lại"""
    img = Image.open(path).convert("RGB").resize(size)
    return img


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
        font_name = os.path.basename(os.path.dirname(chi_path))
        glyph_name = os.path.splitext(os.path.basename(chi_path))[0]

        content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
        style_path = os.path.join(args.english_dir, font_name, "A+.png")

        if not (os.path.exists(content_path) and os.path.exists(style_path)):
            continue

        samples.append({
            "content": content_path,
            "style": style_path,
            "target": chi_path,
            "font": font_name,
            "glyph": glyph_name
        })

    print(f"Tổng số mẫu hợp lệ: {len(samples)}")

    for s in samples:
        font_name, glyph_name = s["font"], s["glyph"]
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
        out_pil = Image.fromarray(
            ((out_img / 2 + 0.5).clamp(0, 1)
             .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ) if isinstance(out_img, torch.Tensor) else out_img

        # Tên file như yêu cầu
        gen_filename = f"{font_name}_{glyph_name}_generated_images.png"
        gt_filename = f"{font_name}_{glyph_name}_gt_images.png"

        # Lưu ảnh generated và groundtruth
        save_single_image(args.save_dir, out_pil, gen_filename)
        target_pil = load_image_tensor(target_path, args.content_image_size)
        save_single_image(args.save_dir, target_pil, gt_filename)

    print(f"\n✅ Hoàn tất inference, ảnh lưu trong: {args.save_dir}")


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
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, f"{args.name}_all_{datetime.now():%H-%M-%S_%d-%m}")
    os.makedirs(args.save_dir, exist_ok=True)
    args.style_image_size = args.content_image_size = (96, 96)

    batch_sampling(args)


if __name__ == "__main__":
    main()
