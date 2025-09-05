import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Training config for FontDiffuser.")
    ################# Experience #################
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--experience_name", type=str, default="fontdiffuer_training")
    parser.add_argument("--data_root", type=str, default=None, help="The font dataset root path.")
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--logging_dir", type=str, default="logs")

    # Model
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--unet_channels", type=tuple, default=(64, 128, 256, 512))
    parser.add_argument("--style_image_size", type=int, default=96)
    parser.add_argument("--content_image_size", type=int, default=96)
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3)
    parser.add_argument("--channel_attn", type=bool, default=True)
    parser.add_argument("--content_start_channel", type=int, default=64)
    parser.add_argument("--style_start_channel", type=int, default=64)
    
    # Training
    parser.add_argument("--unet_type", type=str, default="unet", choices=["unet", "sd3"])
    parser.add_argument("--lang_mode", type=str, default="cross", choices=["cross", "same"])
    parser.add_argument("--same_ratio", type=float, default=0.5)

    parser.add_argument("--phase_2", action="store_true", help="Training in phase 2 using SCR module.")
    parser.add_argument("--phase_1_ckpt_dir", type=str, default=None)
    
    ## SCR
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mode", type=str, default="refinement")
    parser.add_argument("--scr_image_size", type=int, default=96)
    parser.add_argument("--scr_ckpt_path", type=str, default=None)
    parser.add_argument("--num_neg", type=int, default=16, help="Number of negative samples.")
    parser.add_argument("--nce_layers", type=str, default='0,1,2,3')
    parser.add_argument("--sc_coefficient", type=float, default=0.01)

    # TODO: Thêm mới cho cross-lingual SCR
    parser.add_argument("--scr_mode", type=str, default="intra", choices=["intra", "cross", "both"],
                        help="Chọn loại SCR loss: intra (gốc), cross (cross-lingual), both (kết hợp).")
    parser.add_argument("--alpha_intra", type=float, default=1.0,
                        help="Trọng số cho intra-loss khi scr_mode=both.")
    parser.add_argument("--beta_cross", type=float, default=1.0,
                        help="Trọng số cho cross-loss khi scr_mode=both.")

    ## train batch size
    parser.add_argument("--train_batch_size", type=int, default=4)
    ## loss coefficient
    parser.add_argument("--perceptual_coefficient", type=float, default=0.01)
    parser.add_argument("--offset_coefficient", type=float, default=0.5)
    ## step
    parser.add_argument("--max_train_steps", type=int, default=440000)
    parser.add_argument("--ckpt_interval", type=int, default=40000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=100)
    ## learning rate
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=10000)
    ## classifier-free
    parser.add_argument("--drop_prob", type=float, default=0.1)
    ## scheduler
    parser.add_argument("--beta_scheduler", type=str, default="scaled_linear")
    ## optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    
    # Sampling
    parser.add_argument("--algorithm_type", type=str, default="dpmsolver++")
    parser.add_argument("--guidance_type", type=str, default="classifier-free")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--model_type", type=str, default="noise")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--skip_type", type=str, default="time_uniform")
    parser.add_argument("--method", type=str, default="multistep")
    parser.add_argument("--correcting_x0_fn", type=str, default=None)
    parser.add_argument("--t_start", type=str, default=None)
    parser.add_argument("--t_end", type=str, default=None)
    
    parser.add_argument("--local_rank", type=int, default=-1)
    
    return parser