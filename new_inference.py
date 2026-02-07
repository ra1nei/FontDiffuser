import os
import random, string
import torch
from PIL import Image
from datetime import datetime
from torchvision import transforms
import numpy as np
from sample import load_fontdiffuer_pipeline
from utils import save_image_with_content_style
from tqdm import tqdm

def preprocess_image(path, size, device="cuda"):
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)[None, :].to(device)


def collect_images(root_dir):
    """Thu thập toàn bộ ảnh .png, .jpg, .jpeg trong thư mục (bao gồm subfolder)"""
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ] if os.path.exists(root_dir) else []


def save_single_image(save_dir, image, filename):
    """Lưu ảnh PIL vào thư mục"""
    os.makedirs(save_dir, exist_ok=True)
    image.save(os.path.join(save_dir, filename))


def load_image_tensor(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size)
    return img

def save_image_with_content_style(
    save_dir,
    gen_image_pil,
    content_image_pil=None,
    content_image_path=None,
    style_image_path=None,
    resolution=(128, 128),
    filename="out_with_cs.jpg"
):
    os.makedirs(save_dir, exist_ok=True)
    W, H = resolution

    # ----- load content -----
    if content_image_pil is not None:
        content = content_image_pil.resize((W, H))
    else:
        content = Image.open(content_image_path).convert("RGB").resize((W, H))

    # ----- load style -----
    style = Image.open(style_image_path).convert("RGB").resize((W, H))

    # ----- generated -----
    gen = gen_image_pil.resize((W, H))

    # ----- create merged canvas -----
    merged = Image.new("RGB", (W * 3, H))
    merged.paste(content, (0, 0))
    merged.paste(style, (W, 0))
    merged.paste(gen, (W * 2, 0))

    save_path = os.path.join(save_dir, filename)
    merged.save(save_path)
    return save_path

def batch_sampling(args):
    pipe = load_fontdiffuer_pipeline(args)
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(123)

    samples = []

    if args.direction == "e2c":
        print("Mode: English to Chinese (e2c)")
        chinese_images = collect_images(args.chinese_dir)
        print(f"Tổng số ảnh Chinese targets: {len(chinese_images)}")

        for chi_path in chinese_images:
            font_name = os.path.basename(os.path.dirname(chi_path))
            glyph_name = os.path.splitext(os.path.basename(chi_path))[0]

            content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
            
            style_dir = os.path.join(args.english_dir, font_name)
            
            if args.random_style:
                if args.random_mode == "full":
                    candidates = [chr(c)  + "+" for c in range(ord('A'), ord('Z')+1)] + \
                                 [chr(c)  + "+" for c in range(ord('a'), ord('z')+1)]
                elif args.random_mode == "upper":
                    candidates = [chr(c) + "+" for c in range(ord('A'), ord('Z')+1)]
                
                style_candidates = [
                    f for f in os.listdir(style_dir)
                    if os.path.splitext(f)[0] in candidates
                ]
                if not style_candidates: continue
                style_file = random.choice(style_candidates)
            else:
                if args.fixed_style == "A+": style_file = "A+.png"
                elif args.fixed_style == "a": style_file = "a.png"
                else: style_file = "A+.png"

            style_path = os.path.join(style_dir, style_file)

            if not (os.path.exists(content_path) and os.path.exists(style_path)):
                continue

            samples.append({
                "content": content_path,
                "style": style_path,
                "target": chi_path,
                "font": font_name,
                "glyph": glyph_name
            })

    elif args.direction == "c2e":
        print("Mode: Chinese to English (c2e)")
        print(f"Phase: {args.phase}")
        
        english_images = collect_images(args.english_dir)
        print(f"Tổng số ảnh English targets: {len(english_images)}")

        style_search_root = args.chinese_dir 
        if args.phase == "test_unknown_content":
            if not args.chinese_train_dir:
                raise ValueError("Cần cung cấp --chinese_train_dir cho test_unknown_content")
            print(f"Switch Style Source -> Train Dir: {args.chinese_train_dir}")
            style_search_root = args.chinese_train_dir
        else:
            print(f"Keep Style Source -> Test Dir: {args.chinese_dir}")

        candidate_glyph_set = set()
        
        if args.complexity == "all":
            try:
                first_font = os.listdir(style_search_root)[0]
                candidate_glyph_set = set(os.listdir(os.path.join(style_search_root, first_font)))
            except:
                print("Không tìm thấy font nào trong style root")
                return
        else:
            if not args.complexity_root:
                raise ValueError("Vui lòng cung cấp --complexity_root")
            
            complexity_folder_map = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
            ref_folder = os.path.join(args.complexity_root, complexity_folder_map[args.complexity])
            
            if os.path.exists(ref_folder):
                candidate_glyph_set = set([f for f in os.listdir(ref_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Tìm thấy {len(candidate_glyph_set)} glyph mẫu cho độ khó '{args.complexity}'")
        if len(candidate_glyph_set) == 0:
            print("Không tìm thấy glyph mẫu nào!")
            return

        font_valid_glyphs_cache = {}

        for eng_path in english_images:
            font_name = os.path.basename(os.path.dirname(eng_path))
            glyph_name = os.path.splitext(os.path.basename(eng_path))[0]

            if glyph_name.isupper():
                continue

            content_path = os.path.join(args.source_dir, f"{glyph_name}.png")
            
            chinese_font_dir = os.path.join(style_search_root, font_name)
            
            if not os.path.exists(chinese_font_dir):
                continue
            
            if font_name not in font_valid_glyphs_cache:
                try:
                    actual_files_in_font = set(os.listdir(chinese_font_dir))
                    
                    valid_candidates = list(candidate_glyph_set.intersection(actual_files_in_font))
                    
                    font_valid_glyphs_cache[font_name] = valid_candidates
                except OSError:
                    font_valid_glyphs_cache[font_name] = []

            valid_candidates = font_valid_glyphs_cache[font_name]

            if not valid_candidates:
                continue

            style_filename = random.choice(valid_candidates)
            style_path = os.path.join(chinese_font_dir, style_filename)

            if not os.path.exists(content_path):
                continue
            
            samples.append({
                "content": content_path,
                "style": style_path,
                "target": eng_path,
                "font": font_name,
                "glyph": glyph_name
            })

    print(f"Tổng số mẫu hợp lệ: {len(samples)}")

    for s in tqdm(samples, desc="Running inference", ncols=100):
        font_name, glyph_name = s["font"], s["glyph"]
        content_path, style_path, target_path = s["content"], s["style"], s["target"]

        try:
            content_img = preprocess_image(content_path, args.content_image_size, args.device)
            style_img = preprocess_image(style_path, args.style_image_size, args.device)
        except Exception as e:
            print(f"Error loading images: {e}")
            continue

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

        out_pil = out_pil.resize(args.content_image_size)

        # File Naming
        gen_filename = f"{font_name}|{glyph_name}|generated.png"
        gt_filename = f"{font_name}|{glyph_name}|gt.png"

        # Save Generated
        save_single_image(args.save_dir, out_pil, gen_filename)
        
        # Save GT
        try:
            target_pil = load_image_tensor(target_path, args.content_image_size)
            save_single_image(args.save_dir, target_pil, gt_filename)
        except:
            pass

        # Save Merged
        merged_filename = f"{font_name}|{glyph_name}|merged.png"
        save_image_with_content_style(
            save_dir=args.save_dir,
            gen_image_pil=out_pil,
            content_image_path=content_path,
            style_image_path=style_path,
            resolution=args.content_image_size,
            filename=merged_filename
        )

    print(f"\nHoàn tất inference, ảnh lưu trong: {args.save_dir}")

def main():
    from configs.fontdiffuser import get_parser
    parser = get_parser()
    
    # Path Arguments
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True, help="Folder chứa ảnh Content chuẩn")
    parser.add_argument("--english_dir", type=str, required=True, help="Folder dataset English fonts (TEST)")
    parser.add_argument("--chinese_dir", type=str, required=True, help="Folder dataset Chinese fonts (TEST)")
    parser.add_argument("--chinese_train_dir", type=str, default='/kaggle/working/my_data/FTransGAN/train/chinese', help="Folder dataset Chinese fonts (TRAIN) - dùng cho c2e unknown_content")
    
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/results/")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Logic Control
    parser.add_argument("--direction", type=str, default="e2c", choices=["e2c", "c2e"])
    parser.add_argument("--phase", type=str, default="test_unknown_style", 
                        choices=["test_unknown_content", "test_unknown_style"],
                        help="Chế độ test: Content chưa biết (Seen Font) hoặc Style chưa biết (Unseen Font)")

    # Options for E2C
    parser.add_argument("--random_style", action="store_true")
    parser.add_argument("--random_mode", type=str, default="upper", choices=["full", "upper"])
    parser.add_argument("--fixed_style", type=str, default="A+", choices=["A+", "a"])

    # Options for C2E
    parser.add_argument("--complexity", type=str, default="all", choices=["all", "easy", "medium", "hard"])
    parser.add_argument("--complexity_root", type=str, default=None)

    args = parser.parse_args()

    # Xử lý size
    if isinstance(args.style_image_size, int):
        size = args.style_image_size
        args.style_image_size = (size, size)
    if isinstance(args.content_image_size, int):
        size = args.content_image_size
        args.content_image_size = (size, size)
    
    print(f"--- CONFIG ---")
    print(f"Direction: {args.direction}")
    print(f"Phase: {args.phase}")
    if args.direction == "c2e":
        print(f"Complexity: {args.complexity}")
    print(f"--------------")

    batch_sampling(args)

if __name__ == "__main__":
    main()