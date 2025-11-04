import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from evaluator.ssim import SSIM, MSSSIM
from evaluator.fid import FID
import lpips
import numpy as np


# -----------------------
# Dataset khớp cấu trúc mới
# -----------------------
class SimpleResultDataset(Dataset):
    def __init__(self, result_dir, image_size=(96, 96)):
        self.result_dir = result_dir
        self.image_size = image_size

        self.pairs = []
        all_files = os.listdir(result_dir)
        generated = [f for f in all_files if "|generated_images.png" in f]

        for gen in generated:
            prefix = gen.replace("|generated_images.png", "")
            gt = prefix + "|gt_images.png"
            gt_path = os.path.join(result_dir, gt)
            gen_path = os.path.join(result_dir, gen)
            if os.path.exists(gt_path):
                self.pairs.append((gt_path, gen_path, prefix))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_path, gen_path, key = self.pairs[idx]
        gt_img = Image.open(gt_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")

        gt = self.transform(gt_img)
        gen = self.transform(gen_img)

        return gt, gen, key


# -----------------------
# Evaluation Runner
# -----------------------
def evaluate(result_dir, batch_size=1, device="cuda"):
    dataset = SimpleResultDataset(result_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # metrics
    ssim_fn = SSIM(window_size=11, size_average=True)
    msssim_fn = MSSSIM(size_average=True)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    fid_fn = FID(mode="style", num_classes=1, gpu_ids=[0])

    ssim_scores, msssim_scores, lpips_scores = [], [], []

    # FID cần gom nhiều ảnh (sẽ gom hết rồi tính 1 lần)
    gen_images, gt_images = [], []

    for gt, gen, _ in dataloader:
        gt, gen = gt.to(device), gen.to(device)

        # SSIM & MSSSIM
        ssim_val = ssim_fn(gen, gt).item()
        msssim_val = msssim_fn(gen, gt).item()
        ssim_scores.append(ssim_val)
        msssim_scores.append(msssim_val)

        # LPIPS
        lpips_val = lpips_fn(gen, gt).mean().item()
        lpips_scores.append(lpips_val)

        # Lưu FID input
        gen_images.append(gen.cpu())
        gt_images.append(gt.cpu())

    # ---- Tính FID ----
    gen_images = torch.cat(gen_images, dim=0)
    gt_images = torch.cat(gt_images, dim=0)
    fid_score = fid_fn.compute_fid_given_tensors(gen_images, gt_images)

    # ---- In kết quả ----
    print(f"Evaluated on {len(dataset)} pairs")
    print(f"SSIM:   {np.mean(ssim_scores):.4f}")
    print(f"MSSSIM: {np.mean(msssim_scores):.4f}")
    print(f"LPIPS:  {np.mean(lpips_scores):.4f}")
    print(f"FID:    {fid_score:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    result_folder = f"./results/{args.model}-{args.name}"
    evaluate(result_folder)