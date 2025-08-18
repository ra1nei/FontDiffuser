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
                raise FileNotFoundError(f"❌ Không tìm thấy file content image: {content_image_path} hoặc {fallback_path}")

        # --- quyết định same hay cross ---
        script = self.detect_script(style)
        if random.random() < self.same_ratio:
            # same-lingual
            if script == "latin":
                style_folder = random.choice(self.latin_fonts)
            else:
                style_folder = random.choice(self.chinese_fonts)
        else:
            # cross-lingual
            if script == "latin":
                style_folder = random.choice(self.chinese_fonts)
            else:
                style_folder = random.choice(self.latin_fonts)

        # --- load style + target ---
        style_image = Image.open(random.choice([
            path for path in self.style_to_images[style_folder]
        ])).convert("RGB")

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
        }

        if self.scr:
            # lấy danh sách style khác với style hiện tại
            style_list = [s for s in self.style_to_images if s != style]

            # lọc theo script
            if script == "latin":
                style_list = [s for s in style_list if s.endswith("english")]
            else:
                style_list = [s for s in style_list if s.endswith("chinese")]

            # chỉ giữ các style thực sự có content
            valid_neg_styles = []
            for s in style_list:
                paths = self.style_to_images[s]
                # kiểm tra xem có file +content tồn tại không
                if any(os.path.basename(p).split('+')[-1].split('.')[0].rstrip('+') == content for p in paths):
                    valid_neg_styles.append(s)

            if len(valid_neg_styles) < self.num_neg:
                raise ValueError(f"Không đủ negative images cho glyph {content}")

            # chọn negative images từ valid_neg_styles
            neg_images = []
            for _ in range(self.num_neg):
                neg_style = random.choice(valid_neg_styles)
                valid_neg_styles.remove(neg_style)
                # tìm file chứa content
                neg_paths = [p for p in self.style_to_images[neg_style] if os.path.basename(p).split('+')[-1].split('.')[0].rstrip('+') == content]
                neg_path = neg_paths[0]  # lấy file đầu tiên tìm được
                neg_image = Image.open(neg_path).convert("RGB")
                if self.transforms:
                    neg_image = self.transforms[2](neg_image)
                neg_images.append(neg_image.unsqueeze(0))
            sample["neg_images"] = torch.cat(neg_images, dim=0)

        return sample

    def __len__(self):
        return len(self.target_images)
