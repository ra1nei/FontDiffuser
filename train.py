import os
import math
import time
import logging
import random
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.models import AutoencoderKL

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 SD3AdapterUNet,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler,
                 build_scr)
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)

logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():

    args = get_args()
    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir
    )
    torch.autograd.set_detect_anomaly(True)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Seed
    if args.seed is not None:
        set_seed(args.seed)

    # Build encoders & scheduler
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)

    vae = None
    if args.unet_type == "unet":
        unet = build_unet(args=args)

    elif args.unet_type == "sd3":
        

        from diffusers import StableDiffusion3Pipeline

        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype="float16"
        )
        pipe.to("cuda")

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium",
            subfolder="vae"
        ).to(accelerator.device)
        vae.requires_grad_(False)

    else:
        raise ValueError(f"Unknown unet_type {args.unet_type}")

    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)

    perceptual_loss = ContentPerceptualLoss()

    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # Datasets
    content_transforms = transforms.Compose([
        transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    style_transforms = transforms.Compose([
        transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    target_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # -------------------------
    # font selection
    train_root = os.path.join(args.data_root, "train", "TargetImage")
    all_style_folders = [f for f in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, f))]

    # chia theo ngôn ngữ
    def get_lang(folder_name: str):
        suffix = folder_name.lower().rsplit("_", 1)[-1]  # lấy cụm sau cùng sau dấu "_"
        if suffix in ("chinese", "english"):
            return suffix
        return None

    chinese_folders = [f for f in all_style_folders if get_lang(f) == "chinese"]
    english_folders = [f for f in all_style_folders if get_lang(f) == "english"]

    print(f"Chinese: {len(chinese_folders)}, English: {len(english_folders)}")
    print(f"Total fonts (raw): {len(chinese_folders) + len(english_folders)}")

    # chọn theo mode
    if args.lang_mode == "same":
        selected_style_folders = chinese_folders
    elif args.lang_mode == "cross":
        selected_style_folders = chinese_folders + english_folders
    else:
        raise ValueError(f"Unsupported lang_mode: {args.lang_mode}")

    # thống kê
    n_chinese = sum("chinese" in f.lower() for f in selected_style_folders)
    n_english = sum("english" in f.lower() for f in selected_style_folders)

    print(f"Using {len(selected_style_folders)} folders ({n_chinese} zh, {n_english} en)")
    # -------------------------

    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2
    )
    print(f"Total target images: {len(train_font_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN()
    )
    
    # Optimizer
    if args.scale_lr:
        args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    if args.phase_2:
        scr = scr.to(accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            with accelerator.accumulate(model):
                bsz = target_images.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device).long()

                if args.unet_type == "unet":
                    noise = torch.randn_like(target_images)
                    noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                    noise_pred, offset_out_sum = model(
                        x_t=noisy_target_images, 
                        timesteps=timesteps, 
                        style_images=style_images,
                        content_images=content_images,
                        content_encoder_downsample_size=args.content_encoder_downsample_size)

                elif args.unet_type == "sd3":
                    # latent pipeline
                    latents = vae.encode(target_images).latent_dist.sample() * 0.18215
                    noise = torch.randn_like(latents)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    content_feats = content_encoder(content_images)
                    style_embed = style_encoder(style_images)

                    noise_pred = unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=content_feats,
                    )[0]
                    offset_out_sum = torch.tensor(0.0, device=latents.device)

                # --- Loss ---
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2

                if args.unet_type == "unet":
                    pred_original_sample_norm = x0_from_epsilon(
                        scheduler=noise_scheduler,
                        noise_pred=noise_pred,
                        x_t=noisy_target_images,
                        timesteps=timesteps)
                    pred_original_sample = reNormalize_img(pred_original_sample_norm)

                elif args.unet_type == "sd3":
                    pred_original_sample_norm = x0_from_epsilon(
                        scheduler=noise_scheduler,
                        noise_pred=noise_pred,
                        x_t=noisy_latents,
                        timesteps=timesteps)
                    pred_original_sample = vae.decode(pred_original_sample_norm / 0.18215).sample

                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)

                loss = diff_loss + args.perceptual_coefficient * percep_loss + args.offset_coefficient * offset_loss

                # TODO
                if args.phase_2:
                    # Lấy negative samples
                    neg_images = samples["neg_images"]

                    # --- SCR Mode ---
                    if args.scr_mode == "intra":
                        sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                            pred_original_sample_norm,  # sample
                            target_images,              # intra-positive
                            neg_images,                 # intra-negative
                            nce_layers=args.nce_layers
                        )
                        intra_loss = scr.calculate_nce_loss(sample_style_embeddings, pos_style_embeddings, neg_style_embeddings)
                        sc_loss = intra_loss

                    elif args.scr_mode == "cross":
                        sample_style_embeddings, cross_pos_embeddings, cross_neg_embeddings = scr(
                            pred_original_sample_norm,  # sample
                            samples["cross_pos_images"],  # cross-positive
                            samples["cross_neg_images"],  # cross-negative
                            nce_layers=args.nce_layers
                        )
                        cross_loss = scr.calculate_nce_loss(sample_style_embeddings, cross_pos_embeddings, cross_neg_embeddings)
                        sc_loss = cross_loss

                    elif args.scr_mode == "both":
                        # Intra
                        sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                            pred_original_sample_norm,
                            target_images,
                            neg_images,
                            nce_layers=args.nce_layers
                        )
                        intra_loss = scr.calculate_nce_loss(sample_style_embeddings, pos_style_embeddings, neg_style_embeddings)

                        # Cross
                        sample_style_embeddings, cross_pos_embeddings, cross_neg_embeddings = scr(
                            pred_original_sample_norm,
                            samples["cross_pos_images"],
                            samples["cross_neg_images"],
                            nce_layers=args.nce_layers
                        )
                        cross_loss = scr.calculate_nce_loss(sample_style_embeddings, cross_pos_embeddings, cross_neg_embeddings)

                        sc_loss = args.alpha_intra * intra_loss + args.beta_cross * cross_loss

                    else:
                        raise ValueError(f"❌ Unsupported scr_mode: {args.scr_mode}")

                    # Thêm SCR vào tổng loss
                    loss += args.sc_coefficient * sc_loss


                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    save_dir = f"{args.output_dir}/global_step_{global_step}"
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                    torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                    torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                    torch.save(model, f"{save_dir}/total_model.pth")
                    logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}] Save checkpoint step {global_step}")
                    print("Save checkpoint on step {}".format(global_step))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())}] Step {global_step} => train_loss = {loss}")
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
