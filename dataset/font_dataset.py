import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    return nonorm_transform


class FontDataset(Dataset):
    """The dataset of font generation"""
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg

        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = os.path.basename(target_image_path)

        name = target_image_name.split('.')[0].rstrip('+')  # bỏ dấu '+' nếu có ở cuối
        parts = name.split('+')
        if len(parts) != 2:
            raise ValueError(f"❌ Tên ảnh không hợp lệ (phải có đúng 1 dấu '+'): {target_image_name}")
        style, content = parts

        # Read content image — thử không '+' trước, nếu không có thì thêm lại '+'
        content_base_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        if os.path.exists(content_base_path):
            content_image_path = content_base_path
        else:
            fallback_path = f"{self.root}/{self.phase}/ContentImage/{content}+.jpg"
            if os.path.exists(fallback_path):
                content_image_path = fallback_path
            else:
                raise FileNotFoundError(f"❌ Không tìm thấy file content image: {content_base_path} hoặc {fallback_path}")

        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
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
            style_list = list(self.style_to_images.keys())
            style_list.remove(style)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                style_list.remove(choose_style)
                neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_names.append(neg_name)

            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
