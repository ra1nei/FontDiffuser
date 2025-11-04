import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from .ssim import SSIM, MSSSIM
import lpips
import matplotlib.pyplot as plt
from torch_fidelity import calculate_metrics  # ✅ dùng FID thư viện


class Evaluator():
    def __init__(self, opt, num_classes=None, text2label=None):
        self.text2label = text2label
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out_root = os.path.join(opt.results_dir)
        os.makedirs(self.out_root, exist_ok=True)

        # Metrics
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionSSIM = SSIM().to(self.device)
        self.criterionMSSSIM = MSSSIM(weights=[0.45, 0.3, 0.25]).to(self.device)
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
        """Compute LPIPS perceptual distance (lower = better)"""
        with torch.no_grad():
            dist = self.criterionLPIPS(self.gt_images, self.generated_images)
        self.lpips = dist.mean().item()

    def compute_fid(self):
        """Compute FID using torch_fidelity (requires images saved temporarily)"""
        # Lưu tạm ảnh ra hai thư mục để tính FID
        tmp_real = os.path.join(self.out_root, "_fid_real")
        tmp_fake = os.path.join(self.out_root, "_fid_fake")
        os.makedirs(tmp_real, exist_ok=True)
        os.makedirs(tmp_fake, exist_ok=True)

        gt_img = ((self.gt_images[0] / 2 + 0.5) * 255).cpu().numpy().squeeze().astype(np.uint8)
        gen_img = ((self.generated_images[0] / 2 + 0.5) * 255).cpu().numpy().squeeze().astype(np.uint8)

        Image.fromarray(gt_img).save(os.path.join(tmp_real, f"{self.labels}_gt.png"))
        Image.fromarray(gen_img).save(os.path.join(tmp_fake, f"{self.labels}_gen.png"))

        # Chỉ tính trung bình một lần cuối (toàn batch)
        self.fid = 0.0  # placeholder

    def evaluate(self, data):
        self.set_input(data)
        self.compute_l1()
        self.compute_ssim()
        self.compute_msssim()
        self.compute_lpips()
        self.compute_fid()  # chỉ lưu ảnh tạm cho FID thư viện

    def get_current_results(self):
        return {
            'batch_size': self.gt_images.shape[0],
            'l1': self.l1,
            'ssim': self.ssim,
            'msssim': self.msssim,
            'lpips': self.lpips,
            'fid': self.fid,
        }

    def record_current_results(self):
        print('----------- current results -------------')
        print(f"label       : {self.labels}")
        print(f"batch size  : {self.gt_images.shape[0]}")
        print(f"l1          : {self.l1}")
        print(f"ssim        : {self.ssim}")
        print(f"msssim      : {self.msssim}")
        print(f"lpips       : {self.lpips}")
        print()

        res = [
            f"{self.gt_images.shape[0]}\n",
            f"{self.l1}\n",
            f"{self.ssim}\n",
            f"{self.msssim}\n",
            f"{self.lpips}\n",
            f"{self.fid}\n",
        ]
        with open(os.path.join(self.out_root, self.labels) + '.txt', 'w') as f:
            f.writelines(res)

    def compute_final_results(self):
        # Khi đã lưu toàn bộ ảnh, gọi FID ở đây
        real_dir = os.path.join(self.out_root, "_fid_real")
        fake_dir = os.path.join(self.out_root, "_fid_fake")

        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            print("⚠️ Không có ảnh để tính FID.")
            fid = 0.0
        else:
            print("🧮 Đang tính FID bằng torch_fidelity...")
            metrics = calculate_metrics(
                input1=real_dir,
                input2=fake_dir,
                fid=True,
                cuda=torch.cuda.is_available(),
                verbose=False
            )
            fid = metrics["frechet_inception_distance"]

        # Đọc lại các file txt để tính trung bình
        files = [f for f in os.listdir(self.out_root) if f.endswith(".txt") and f != "final_results.txt"]
        num_images, l1, ssim, msssim, lpips_val = 0, 0, 0, 0, 0

        for file in files:
            with open(os.path.join(self.out_root, file), 'r') as f:
                l = f.read().split('\n')
                if len(l) < 6:
                    continue
                batch_size = int(l[0])
                num_images += batch_size
                l1 += float(l[1]) * batch_size
                ssim += float(l[2]) * batch_size
                msssim += float(l[3]) * batch_size
                lpips_val += float(l[4]) * batch_size

        if num_images == 0:
            print("⚠️ Không có ảnh hợp lệ — bỏ qua thống kê trung bình.")
            return

        l1 /= num_images
        ssim /= num_images
        msssim /= num_images
        lpips_val /= num_images

        res = [
            f"l1:{l1}\n",
            f"ssim:{ssim}\n",
            f"msssim:{msssim}\n",
            f"lpips:{lpips_val}\n",
            f"fid:{fid}\n",
        ]
        with open(os.path.join(self.out_root, 'final_results.txt'), 'w') as f:
            f.writelines(res)
        print(f"✅ results saved at {os.path.join(self.out_root, 'final_results.txt')}")
        print(f"FID: {fid:.4f}")

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
