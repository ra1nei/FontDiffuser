import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False, lang_mode="same"):
        super().__init__()
        self.root = args.data_root
        self.phase = phase # ví dụ: "train_same", "train_cross"
        self.scr = scr
        self.lang_mode = args.lang_mode
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
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
            print("Error")
            return None

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        filename = target_image_name.split('.')[0]
        style_lang, content = filename.split('+', 1)
        style, lang = style_lang.split('_', 1)

        print("Style:", style, "Content:", content)
        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image = Image.open(content_image_path).convert('RGB')

        if self.lang_mode == "same":
            # Same: chọn style khác nhưng cùng content, cùng script (vd: Chinese)
            images_related_style = self.style_to_images[style+'_'+lang].copy()
            images_related_style.remove(target_image_path)
            style_image_path = random.choice(images_related_style)

        elif self.lang_mode == "cross":
            # Cross: chọn style khác nhưng cùng script với content
            if content.isascii():  
                valid_styles = [s for s in self.style_to_images if s.isascii()]
            else:
                valid_styles = [s for s in self.style_to_images if not s.isascii()]

            # Remove current style
            valid_styles = [s for s in valid_styles if s != style]

            choose_style = random.choice(valid_styles)
            style_image_path = random.choice(self.style_to_images[choose_style])

        # Read style_image
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
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
            "nonorm_target_image": nonorm_target_image}
        
        if self.scr:
            # Get neg images from different styles of the same content
            style_list = list(self.style_to_images.keys())
            if style in style_list:
                style_list.remove(style)

            choose_neg_names = []
            for i in range(self.num_neg):
                if not style_list:
                    break
                choose_style = random.choice(style_list)
                style_list.remove(choose_style)

                neg_path1 = f"{self.root}/{self.phase}/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                neg_path2 = f"{self.root}/{self.phase}/TargetImage/{choose_style}/{choose_style}+{content}+.jpg"

                if os.path.exists(neg_path1):
                    choose_neg_names.append(neg_path1)
                elif os.path.exists(neg_path2):
                    choose_neg_names.append(neg_path2)
                else:
                    # bỏ qua nếu font thiếu glyph
                    continue

            neg_images = []
            for neg_name in choose_neg_names:
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                neg_images.append(neg_image[None, :, :, :])

            if len(neg_images) > 0:
                sample["neg_images"] = torch.cat(neg_images, dim=0)
            else:
                # nếu không có negative nào thì raise warning
                raise FileNotFoundError(f"❌ No valid negative images found for content {content}")

        return sample

    def __len__(self):
        return len(self.target_images)