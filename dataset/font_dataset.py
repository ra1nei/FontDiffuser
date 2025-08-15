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
    def __init__(self, args, phase, transforms=None, scr=False, allowed_styles=None):
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

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = os.path.basename(target_image_path)

        # Bỏ dấu '+' ở cuối nếu có
        name = target_image_name.split('.')[0].rstrip('+')

        # Tách style và content từ dấu '+' cuối cùng
        if '+' not in name:
            raise ValueError(f"❌ Tên ảnh không hợp lệ (không có dấu '+'): {target_image_name}")

        last_plus_index = name.rfind('+')
        style = name[:last_plus_index]
        content = name[last_plus_index + 1:]

        if not content:
            raise ValueError(f"❌ Tên ảnh bị thiếu content glyph: {target_image_name}")

        # Tìm content image tương ứng (ưu tiên không có '+' trong tên file)
        content_dir = os.path.join(self.root, self.phase, "ContentImage")
        content_image_path = os.path.join(content_dir, f"{content}.jpg")
        if not os.path.exists(content_image_path):
            fallback_path = os.path.join(content_dir, f"{content}+.jpg")
            if os.path.exists(fallback_path):
                content_image_path = fallback_path
            else:
                raise FileNotFoundError(f"❌ Không tìm thấy file content image: {content_image_path} hoặc {fallback_path}")

        # Load images
        content_image = Image.open(content_image_path).convert('RGB')
        style_image = Image.open(random.choice([
            path for path in self.style_to_images[style] if path != target_image_path
        ])).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
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
            # Lấy ảnh "neg" khác style nhưng cùng content
            style_list = [s for s in self.style_to_images if s != style]
            neg_images = []
            for _ in range(self.num_neg):
                neg_style = random.choice(style_list)
                style_list.remove(neg_style)
                neg_path = os.path.join(self.root, "train", "TargetImage", neg_style, f"{neg_style}+{content}.jpg")
                if not os.path.exists(neg_path):
                    alt_path = os.path.join(self.root, "train", "TargetImage", neg_style, f"{neg_style}+{content}+.jpg")
                    if os.path.exists(alt_path):
                        neg_path = alt_path
                    else:
                        raise FileNotFoundError(f"❌ Không tìm thấy negative image: {neg_path} hoặc {alt_path}")
                neg_image = Image.open(neg_path).convert("RGB")
                if self.transforms:
                    neg_image = self.transforms[2](neg_image)
                neg_images.append(neg_image.unsqueeze(0))
            sample["neg_images"] = torch.cat(neg_images, dim=0)

        return sample

    def __len__(self):
        return len(self.target_images)
