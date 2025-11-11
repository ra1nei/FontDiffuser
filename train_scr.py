# train_scr.py
import os
import argparse
import math
from tqdm.auto import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import set_seed

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from src.build import build_scr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--phase", type=str, default="train")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lang_mode", type=str, default="same", choices=["same", "cross"])

    # SCR-specific
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mode", type=str, default="training", choices=["training", "inference"])
    parser.add_argument("--loss_mode", type=str, default="intra", choices=["intra", "cross", "both"])
    parser.add_argument("--nce_layers", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--alpha_intra", type=float, default=0.3)
    parser.add_argument("--beta_cross", type=float, default=0.7)
    parser.add_argument("--num_neg", type=int, default=4)

    # Logging & checkpoint
    parser.add_argument("--save_dir", type=str, default="./scr_checkpoints")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--ckpt_interval", type=int, default=20000)
    parser.add_argument("--max_train_steps", type=int, default=200000)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--report_to", type=str, default="wandb", help="Logging backend (wandb/tensorboard/none)")
    parser.add_argument("--experience_name", type=str, default="SCR-Training")
    parser.add_argument("--seed", type=int, default=123)

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


def save_checkpoint(path, model, optimizer, step):
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"[INFO] Saved checkpoint: {path}")


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    step = ckpt.get("step", 0)
    print(f"[INFO] Resumed training from {path} (global_step={step})")
    return step

def train():
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator(log_with=args.report_to, project_dir=args.save_dir)
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        accelerator.init_trackers(args.experience_name, config=vars(args))

    # transforms & dataloader
    content_tf, style_tf, target_tf = get_transforms(args.resolution)
    dataset = FontDataset(
        args=args,
        phase=args.phase,
        transforms=[content_tf, style_tf, target_tf],
        scr=True,
        scr_mode=args.loss_mode,
        lang_mode=args.lang_mode
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CollateFN(),
        drop_last=True
    )

    # model + optimizer
    scr_model = build_scr(args)
    optimizer = optim.Adam(
        list(scr_model.StyleFeatExtractor.parameters()) + list(scr_model.StyleFeatProjector.parameters()),
        lr=args.learning_rate
    )

    scr_model, optimizer, dataloader = accelerator.prepare(scr_model, optimizer, dataloader)

    global_step = 0
    if args.resume_ckpt and os.path.isfile(args.resume_ckpt):
        global_step = load_checkpoint(args.resume_ckpt, scr_model, optimizer, accelerator.device)

    total_steps = min(args.max_train_steps, args.epochs * len(dataloader))
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, initial=global_step)
    progress_bar.set_description("Training Steps")

    scr_model.train()
    train_loss = 0.0

    while global_step < total_steps:
        for batch in dataloader:
            style = batch["style_image"].to(accelerator.device)
            intra_pos = batch.get("intra_pos_image")
            cross_pos = batch.get("cross_pos_image")
            intra_neg = batch.get("intra_neg_images")
            cross_neg = batch.get("cross_neg_images")

            if intra_pos is not None: intra_pos = intra_pos.to(accelerator.device)
            if cross_pos is not None: cross_pos = cross_pos.to(accelerator.device)
            if intra_neg is not None: intra_neg = intra_neg.to(accelerator.device)
            if cross_neg is not None: cross_neg = cross_neg.to(accelerator.device)

            with accelerator.accumulate(scr_model):
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

                # Tính loss trung bình across processes (nếu distributed)
                avg_loss = accelerator.gather(loss.repeat(args.batch_size)).mean()
                train_loss += avg_loss.item() / accelerator.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()

            # Nếu đã step optimizer thật sự
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0  # reset sau khi log
                progress_bar.update(1)
                progress_bar.set_postfix({"step_loss": loss.item()})

                if global_step % args.ckpt_interval == 0 and accelerator.is_main_process:
                    save_checkpoint(os.path.join(args.save_dir, f"scr_{global_step}.pth"),
                                    accelerator.unwrap_model(scr_model),
                                    optimizer, global_step)

            if global_step >= total_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    train()