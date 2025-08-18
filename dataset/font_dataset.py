import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])


class FontDataset(Dataset):
    """The dataset of font generation"""
    def __init__(self, args, phase, transforms=None, scr=False, allowed_styles=None, same_ratio=0.5):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        self.allowed_styles = allowed_styles
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        # chia nhóm latin vs chinese
        self.latin_fonts = [f for f in allowed_styles if f.endswith("english")]
        self.chinese_fonts = [f for f in allowed_styles if f.endswith("chinese")]

        # tỷ lệ same-lingual
        self.same_ratio = same_ratio

    def get_path(self):
        self.target_images = []
        self.style_to_images = {}
        target_image_dir = os.path.join(self.root, self.phase, "TargetImage")
        for style in os.listdir(target_image_dir):
            if self.allowed_styles is not None and style not in self.allowed_styles:
                continue
            images_related_style = []
            style_dir = os.path.join(target_image_dir, style)
            for img in os.listdir(style_dir):
                img_path = os.path.join(style_dir, img)
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def detect_script(self, style_name: str):
        """Trả về 'latin' hoặc 'chinese' dựa trên tên folder style"""
        if style_name.endswith("english"):
            return "latin"
        elif style_name.endswith("chinese"):
            return "chinese"
        else:
            raise ValueError(f"❌ Không nhận diện được script trong style: {style_name}")

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = os.path.basename(target_image_path)

        # Bỏ dấu '+' ở cuối nếu có
        name = target_image_name.split('.')[0].rstrip('+')

        if '+' not in name:
            raise ValueError(f"❌ Tên ảnh không hợp lệ (không có dấu '+'): {target_image_name}")

        last_plus_index = name.rfind('+')
        style = name[:last_plus_index]
        content = name[last_plus_index + 1:]

        if not content:
            raise ValueError(f"❌ Tên ảnh bị thiếu content glyph: {target_image_name}")

        # --- lấy content image ---
        content_dir = os.path.join(self.root, self.phase, "ContentImage")
        content_image_path = os.path.join(content_dir, f"{content}.jpg")
        if not os.path.exists(content_image_path):
            fallback_path = os.path.join(content_dir, f"{content}+.jpg")
            if os.path.exists(fallback_path):
                content_image_path = fallback_path
            else:
                return self.__getitem__(random.randint(0, len(self.target_images) - 1))
            
        # --- quyết định same hay cross ---
        script = self.detect_script(style)
        if random.random() < self.same_ratio:
            # same-lingual
            style_folder = random.choice(self.latin_fonts if script == "latin" else self.chinese_fonts)
        else:
            # cross-lingual
            style_folder = random.choice(self.chinese_fonts if script == "latin" else self.latin_fonts)

        # --- load style + target ---
        style_image = Image.open(random.choice(self.style_to_images[style_folder])).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
        content_image = Image.open(content_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "neg_images": None,   # mặc định None
        }

        # --- xử lý negative sample ---
        if self.scr:
            style_list = [s for s in self.style_to_images if s != style]

            # lọc same / cross
            if random.random() < self.same_ratio:  # same-lingual
                style_list = [s for s in style_list if (s.endswith("english") if script == "latin" else s.endswith("chinese"))]
            else:  # cross-lingual
                style_list = [s for s in style_list if (s.endswith("chinese") if script == "latin" else s.endswith("english"))]

            # lọc style có content tồn tại
            valid_style_list = []
            for s in style_list:
                path1 = os.path.join(self.root, self.phase, "TargetImage", s, f"{s}+{content}.jpg")
                path2 = os.path.join(self.root, self.phase, "TargetImage", s, f"{s}+{content}+.jpg")
                if os.path.exists(path1) or os.path.exists(path2):
                    valid_style_list.append(s)

            # nếu không có negative hợp lệ → bỏ qua
            if len(valid_style_list) > 0:
                neg_images = []
                for _ in range(min(self.num_neg, len(valid_style_list))):
                    neg_style = random.choice(valid_style_list)
                    valid_style_list.remove(neg_style)
                    neg_path = os.path.join(self.root, self.phase, "TargetImage", neg_style, f"{neg_style}+{content}.jpg")
                    if not os.path.exists(neg_path):
                        neg_path = os.path.join(self.root, self.phase, "TargetImage", neg_style, f"{neg_style}+{content}+.jpg")
                    if os.path.exists(neg_path):
                        try:
                            neg_image = Image.open(neg_path).convert("RGB")
                            if self.transforms:
                                neg_image = self.transforms[2](neg_image)
                            neg_images.append(neg_image.unsqueeze(0))
                        except Exception as e:
                            print(f"⚠️ Bỏ qua neg image lỗi: {neg_path}, {e}")
                            continue

                if len(neg_images) > 0:
                    sample["neg_images"] = torch.cat(neg_images, dim=0)

        return sample

    def __len__(self):
        return len(self.target_images)