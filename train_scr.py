# train_scr.py
import os
import argparse
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from src.build import build_scr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # SCR-specific arguments
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss_mode", type=str, default="intra", choices=["intra", "cross", "both"])
    parser.add_argument("--nce_layers", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--alpha_intra", type=float, default=0.3)
    parser.add_argument("--beta_cross", type=float, default=0.7)
    parser.add_argument("--num_neg", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="./scr_checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_transforms(resolution):
    return (
        transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )

def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # transforms
    content_tf, style_tf, target_tf = get_transforms(args.resolution)

    # dataset & dataloader
    dataset = FontDataset(
        args=args,
        phase=args.phase,
        transforms=[content_tf, style_tf, target_tf],
        scr=True,
        scr_mode=args.loss_mode,
        lang_mode="same"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CollateFN(),
        drop_last=True
    )

    # model SCR
    scr_model = build_scr(args)  # dùng đúng build_scr trong src
    device = torch.device(args.device)
    scr_model = scr_model.to(device)
    scr_model.train()

    optimizer = optim.Adam(
        list(scr_model.StyleFeatExtractor.parameters()) + list(scr_model.StyleFeatProjector.parameters()),
        lr=args.learning_rate
    )

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            style = batch["style_image"].to(device)
            intra_pos = batch.get("intra_pos_image", None)
            cross_pos = batch.get("cross_pos_image", None)
            intra_neg = batch.get("intra_neg_images", None)
            cross_neg = batch.get("cross_neg_images", None)

            if intra_pos is not None: intra_pos = intra_pos.to(device)
            if cross_pos is not None: cross_pos = cross_pos.to(device)
            if intra_neg is not None: intra_neg = intra_neg.to(device)
            if cross_neg is not None: cross_neg = cross_neg.to(device)

            optimizer.zero_grad()
            sample_s, intra_pos_s, cross_pos_s, intra_neg_s, cross_neg_s = scr_model(
                sample_imgs=style,
                intra_pos_imgs=intra_pos,
                cross_pos_imgs=cross_pos,
                intra_neg_imgs=intra_neg,
                cross_neg_imgs=cross_neg,
                nce_layers=args.nce_layers
            )
            loss = scr_model.calculate_nce_loss(
                sample_s,
                intra_pos_s=intra_pos_s,
                cross_pos_s=cross_pos_s,
                intra_neg_s=intra_neg_s,
                cross_neg_s=cross_neg_s
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # save checkpoint cuối mỗi epoch
        ckpt_path = os.path.join(args.save_dir, f"scr_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "step": global_step,
            "model_state": scr_model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    train()