import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from .ssim import SSIM, MSSSIM
from .fid import FID
from .classifier import Classifier
import lpips
import matplotlib.pyplot as plt


class Evaluator():
    def __init__(self, opt, num_classes=None, text2label=None):
        self.text2label = text2label
        self.evaluate_mode = opt.evaluate_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out_root = os.path.join(opt.results_dir)
        # Metrics
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSSIM = SSIM().to(self.device)
        self.criterionMSSSIM = MSSSIM(weights=[0.45, 0.3, 0.25]).to(self.device)
        self.criterionFID = FID(opt.evaluate_mode, num_classes, gpu_ids=opt.gpu_ids)
        self.criterionLPIPS = lpips.LPIPS(net='vgg').to(self.device)  # ✅ new metric

    def set_input(self, data):
        self.gt_images = data[0].permute([1, 0, 2, 3]).to(self.device)
        self.generated_images = data[1].permute([1, 0, 2, 3]).to(self.device)
        self.labels = data[2][0]

    def compute_l1(self):
        self.l1 = self.criterionL1(self.gt_images / 2 + 0.5, self.generated_images / 2 + 0.5).item()

    def compute_ssim(self):
        self.ssim = self.criterionSSIM(self.gt_images / 2 + 0.5, self.generated_images / 2 + 0.5).item()

    def compute_msssim(self):
        self.msssim = self.criterionMSSSIM(self.gt_images / 2 + 0.5, self.generated_images / 2 + 0.5).item()
        if np.isnan(self.msssim):
            self.msssim = 0.0

    def compute_lpips(self):
        """Compute LPIPS perceptual distance (lower = more similar)"""
        with torch.no_grad():
            dist = self.criterionLPIPS(self.gt_images, self.generated_images)
        self.lpips = dist.mean().item()

    def compute_acc(self):
        if self.evaluate_mode == 'content':
            labels = self.text2label[self.labels + '.png']
        else:
            labels = self.text2label[self.labels]
        self.acc = self.criterionFID.get_acc(labels).item()

    def compute_fid(self):
        self.fid = self.criterionFID.forward(self.gt_images, self.generated_images)

    def evaluate(self, data):
        self.set_input(data)
        self.compute_fid()
        self.compute_acc()
        self.compute_l1()
        self.compute_ssim()
        self.compute_msssim()
        self.compute_lpips()  # ✅ include LPIPS

    def get_current_results(self):
        return {
            'batch_size': self.gt_images.shape[0],
            'l1': self.l1,
            'ssim': self.ssim,
            'msssim': self.msssim,
            'lpips': self.lpips,  # ✅ added
            'fid': self.fid,
            'num_correct': self.acc,
        }

    def record_current_results(self):
        print('----------- current results -------------')
        print(f"label       : {self.labels}")
        print(f"batch size  : {self.gt_images.shape[0]}")
        print(f"num_correct : {self.acc}")
        print(f"l1          : {self.l1}")
        print(f"ssim        : {self.ssim}")
        print(f"msssim      : {self.msssim}")
        print(f"lpips       : {self.lpips}")
        print(f"fid         : {self.fid}")
        print()

        res = [
            f"{self.gt_images.shape[0]}\n",
            f"{self.acc}\n",
            f"{self.l1}\n",
            f"{self.ssim}\n",
            f"{self.msssim}\n",
            f"{self.lpips}\n",
            f"{self.fid}\n",
        ]
        os.makedirs(self.out_root, exist_ok=True)
        with open(os.path.join(self.out_root, self.labels) + '.txt', 'w') as f:
            f.writelines(res)

    def compute_final_results(self):
        num_images, num_correct, l1, ssim, msssim, lpips_val, fid = 0, 0, 0, 0, 0, 0, 0
        files = os.listdir(self.out_root)
        print('loading metrics...')
        for file in files:
            if not file.endswith('.txt') or file == 'final_results.txt':
                continue
            with open(os.path.join(self.out_root, file), 'r') as f:
                l = f.read().split('\n')
                if len(l) < 7:  # skip incomplete
                    continue
                batch_size = int(l[0])
                num_images += batch_size
                num_correct += int(float(l[1]))
                l1 += float(l[2]) * batch_size
                ssim += float(l[3]) * batch_size
                msssim += float(l[4]) * batch_size
                lpips_val += float(l[5]) * batch_size
                fid += float(l[6]) * batch_size

        acc = num_correct / num_images
        l1 /= num_images
        ssim /= num_images
        msssim /= num_images
        lpips_val /= num_images
        fid /= num_images

        res = [
            f"acc:{acc}\n",
            f"l1:{l1}\n",
            f"ssim:{ssim}\n",
            f"msssim:{msssim}\n",
            f"lpips:{lpips_val}\n",
            f"fid:{fid}\n",
        ]
        with open(os.path.join(self.out_root, 'final_results.txt'), 'w') as f:
            f.writelines(res)
        print(f"results saved at {os.path.join(self.out_root, 'final_results.txt')}")

    def show_examples(self):
        idx = np.random.randint(0, self.gt_images.shape[0])
        plt.figure(figsize=[5, 10])
        plt.subplot(1, 2, 1)
        plt.imshow(self.gt_images.cpu()[idx, 0, :, :], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(self.generated_images.cpu()[idx, 0, :, :], cmap='gray')
        plt.axis('off')
        plt.show()


class EvaluatorDataset(Dataset):
    def __init__(self, opt):
        data_root = os.path.join(opt.results_dir, opt.name, f"{opt.phase}_{opt.epoch}", 'images')
        part = 1 if opt.evaluate_mode == 'content' else 0
        all_image_paths = os.listdir(data_root)
        self.all_classes = list(set(path.split('|')[part] for path in all_image_paths))
        self.table = {key: [[], []] for key in self.all_classes}

        for path in all_image_paths:
            key = path.split('|')[part]
            category = path.split('|')[2]
            path = os.path.join(data_root, path)
            if category == 'gt_images.png':
                self.table[key][0].append(path)
            elif category == 'generated_images.png':
                self.table[key][1].append(path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def __getitem__(self, idx):
        key = self.all_classes[idx]
        gt_images = torch.cat([self.load_image(p) for p in sorted(self.table[key][0])], 0)
        generated_images = torch.cat([self.load_image(p) for p in sorted(self.table[key][1])], 0)
        return (gt_images, generated_images, key)

    def __len__(self):
        return len(self.all_classes)

    def load_image(self, path):
        image = Image.open(path).convert('L')
        return self.transform(image)
