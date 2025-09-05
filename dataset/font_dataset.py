import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_nonorm_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])


class FontDataset(Dataset):
    """Dataset cho font generation + SCR (intra/cross/both)."""
    def __init__(self, args, phase, transforms=None, scr=False, scr_mode="intra"):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        self.scr_mode = scr_mode  # 'intra', 'cross', 'both'
        if self.scr:
            self.num_neg = args.num_neg

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"

        styles_to_process = os.listdir(target_image_dir)
        for style in styles_to_process:
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def get_script(self, name: str):
        if name.endswith("_chinese"):
            return "chinese"
        elif name.endswith("_english"):
            return "latin"
        else:
            return None

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        filename = os.path.splitext(os.path.basename(target_image_path))[0]

        # Parse style, lang, content
        last_plus_index = filename.rfind('+')
        style_lang_part = filename[:last_plus_index]
        content = filename[last_plus_index + 1:]
        last_underscore_index = style_lang_part.rfind('_')
        style = style_lang_part[:last_underscore_index]
        lang = style_lang_part[last_underscore_index:]
        script = self.get_script(style + lang)

        # Load content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        # Ground-truth target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Chọn style image (giữ nguyên logic cũ)
        style_image_path = random.choice(self.style_to_images[style + lang])
        style_image = Image.open(style_image_path).convert("RGB")

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image
        }

        if self.scr:
            # === Intra Positive ===
            if self.scr_mode in ["intra", "both"]:
                intra_pos_path = target_image_path  # chính ground-truth
                intra_pos_image = Image.open(intra_pos_path).convert("RGB")
                if self.transforms is not None:
                    intra_pos_image = self.transforms[2](intra_pos_image)
                sample["intra_pos_image"] = intra_pos_image

            # === Cross Positive ===
            if self.scr_mode in ["cross", "both"]:
                cross_style = style + ("_english" if script == "chinese" else "_chinese")
                if cross_style in self.style_to_images:
                    cross_pos_path = random.choice(self.style_to_images[cross_style])
                    cross_pos_image = Image.open(cross_pos_path).convert("RGB")
                    if self.transforms is not None:
                        cross_pos_image = self.transforms[2](cross_pos_image)
                    sample["cross_pos_image"] = cross_pos_image

            # === Intra Negatives ===
            if self.scr_mode in ["intra", "both"]:
                intra_neg_images = []
                for neg_style in self.style_to_images:
                    if neg_style != style + lang and neg_style.endswith(lang):
                        neg_candidates = [p for p in self.style_to_images[neg_style] if p.endswith("+" + content + ".jpg")]
                        if len(neg_candidates) > 0:
                            neg_path = random.choice(neg_candidates)
                            neg_image = Image.open(neg_path).convert("RGB")
                            if self.transforms is not None:
                                neg_image = self.transforms[2](neg_image)
                            intra_neg_images.append(neg_image[None, :, :, :])
                if len(intra_neg_images) > 0:
                    sample["intra_neg_images"] = torch.cat(intra_neg_images, dim=0)

            # === Cross Negatives ===
            if self.scr_mode in ["cross", "both"]:
                cross_neg_images = []
                for neg_style in self.style_to_images:
                    if not neg_style.endswith(lang):  # khác script
                        neg_path = random.choice(self.style_to_images[neg_style])
                        neg_image = Image.open(neg_path).convert("RGB")
                        if self.transforms is not None:
                            neg_image = self.transforms[2](neg_image)
                        cross_neg_images.append(neg_image[None, :, :, :])
                if len(cross_neg_images) > 0:
                    sample["cross_neg_images"] = torch.cat(cross_neg_images, dim=0)

        return sample

    def __len__(self):
        return len(self.target_images)