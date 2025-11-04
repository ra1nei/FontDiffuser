import os
import argparse
import torch
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure, FrechetInceptionDistance

# ======================
#  Parse Arguments
# ======================
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="experiment name, e.g., p1_cross-SFUC")
parser.add_argument("--model", type=str, required=True, help="model name, e.g., 851CHIKARA-DZUYOKU-kanaB-2")
args = parser.parse_args()

result_folder = f"./results/{args.model}-{args.name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
#  Define Metrics
# ======================
lpips_metric = lpips.LPIPS(net='vgg').to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
fid_metric = FrechetInceptionDistance(feature=64).to(device)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)

# ======================
#  Collect Image Pairs
# ======================
pairs = []
for f in os.listdir(result_folder):
    if f.endswith("generated_images.png"):
        base = f.replace("generated_images.png", "")
        gt_path = os.path.join(result_folder, base + "gt_images.png")
        gen_path = os.path.join(result_folder, f)
        if os.path.exists(gt_path):
            pairs.append((gen_path, gt_path))

print(f"Found {len(pairs)} image pairs in {result_folder}")

# ======================
#  Evaluate
# ======================
l1_scores, ssim_scores, lpips_scores = [], [], []

for gen_path, gt_path in tqdm(pairs, desc="Evaluating"):
    gen = load_image(gen_path)
    gt = load_image(gt_path)

    # L1
    l1 = torch.mean(torch.abs(gen - gt)).item()
    l1_scores.append(l1)

    # SSIM
    ssim = ssim_metric(gen, gt).item()
    ssim_scores.append(ssim)

    # LPIPS
    lp = lpips_metric(gen, gt).item()
    lpips_scores.append(lp)

    # FID (1 batch = 1 cặp gen/gt)
    gen_uint8 = (gen * 255).clamp(0, 255).to(torch.uint8)
    gt_uint8 = (gt * 255).clamp(0, 255).to(torch.uint8)
    fid_metric.update(gen_uint8, real=False)
    fid_metric.update(gt_uint8, real=True)


fid_value = fid_metric.compute().item()

# ======================
#  Print & Save Results
# ======================
results_text = (
    f"Model: {args.model}\n"
    f"Experiment: {args.name}\n"
    f"Total pairs: {len(pairs)}\n\n"
    f"L1   : {np.mean(l1_scores):.6f}\n"
    f"SSIM : {np.mean(ssim_scores):.6f}\n"
    f"LPIPS: {np.mean(lpips_scores):.6f}\n"
    f"FID  : {fid_value:.6f}\n"
)

print("\n=== RESULTS ===")
print(results_text)

# Lưu ra file
os.makedirs("./metrics", exist_ok=True)
out_path = f"./metrics/{args.model}-{args.name}.txt"
with open(out_path, "w") as f:
    f.write(results_text)

print(f"Metrics saved to: {out_path}")
