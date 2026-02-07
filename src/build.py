from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import (ContentEncoder, 
                 StyleEncoder, 
                 UNet,
                 SCR)

def build_unet(args):
    # --- RSI ---
    if not args.rsi:
        # RSI OFF
        up_blocks = ('UpBlock2D', 'StyleOnlyUpBlock2D', 'StyleOnlyUpBlock2D', 'UpBlock2D')
        print("[Model Config] RSI: OFF")
        print(up_blocks)
    else:
        # RSI ON
        up_blocks = ('UpBlock2D', 'StyleRSIUpBlock2D', 'StyleRSIUpBlock2D', 'UpBlock2D')
        print("[Model Config] RSI: ON")
        print(up_blocks)

    # --- MCA ---
    if not args.mca:
        # MCA OFF
        down_blocks = ('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D')
        print("[Model Config] MCA: OFF")
        print(down_blocks)
    else:
        # MCA ON
        down_blocks = ('DownBlock2D', 'MCADownBlock2D', 'MCADownBlock2D', 'DownBlock2D')
        print("[Model Config] MCA: ON")
        print(down_blocks)


    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=down_blocks,
        up_block_types=up_blocks,
        block_out_channels=args.unet_channels, 
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32,
        deformation_scale=args.deformation_scale,
    )
    
    return unet

def build_style_encoder(args):
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0])
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0])
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_scr(args):
    disable_augment = getattr(args, "disable_scr_augment", False)

    scr = SCR(
        temperature=args.temperature,
        mode=args.mode,
        image_size=args.resolution,
        augment=not disable_augment)
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler