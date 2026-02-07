import torch
from PIL import Image

from .dpm_solver_pytorch import (NoiseScheduleVP, 
                                model_wrapper, 
                                DPM_Solver)

class FontDiffuserDPMPipeline():
    """FontDiffuser pipeline with DPM_Solver scheduler.
    """
    
    def __init__(
        self, 
        model, 
        ddpm_train_scheduler,
        version="V3",
        model_type="noise",
        guidance_type="classifier-free",
        guidance_scale=7.5
    ):
        super().__init__()
        self.model = model
        self.train_scheduler_betas = ddpm_train_scheduler.betas
        # Define the noise schedule
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.train_scheduler_betas)

        self.version = version
        self.model_type = model_type
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale

    def numpy_to_pil(self, images):
        """Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def generate(
        self,
        content_images,
        style_images,
        batch_size,
        order,
        num_inference_step,
        content_encoder_downsample_size,
        t_start=None,
        t_end=None,
        dm_size=(96, 96),
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
        generator=None,
    ):
        if isinstance(dm_size, int):
            dm_size = (dm_size, dm_size)
        model_kwargs = {}
        model_kwargs["version"] = self.version
        model_kwargs["content_encoder_downsample_size"] = content_encoder_downsample_size

        cond = []
        cond.append(content_images)
        cond.append(style_images)

        uncond = []
        uncond_content_images = torch.ones_like(content_images).to(self.model.device)
        uncond_style_images = torch.ones_like(style_images).to(self.model.device)
        uncond.append(uncond_content_images)
        uncond.append(uncond_style_images)

        # 2.Convert the discrete-time model to the continuous-time
        model_fn = model_wrapper(
            model=self.model,
            noise_schedule=self.noise_schedule,
            model_type=self.model_type,
            model_kwargs=model_kwargs,
            guidance_type=self.guidance_type,
            condition=cond, 
            unconditional_condition=uncond,
            guidance_scale=self.guidance_scale
        )

        # 3. Define dpm-solver and sample by multistep DPM-Solver.
        # (We recommend multistep DPM-Solver for conditional sampling)
        # You can adjust the `steps` to balance the computation costs and the sample quality.
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=self.noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn
        )
        # If the DPM is defined on pixel-space images, you can further set `correcting_x0_fn="dynamic_thresholding"

        # 4. Generate
        # Sample gaussian noise to begin loop => [batch, 3, height, width]
        x_T = torch.randn(
            (batch_size, 3, dm_size[0], dm_size[1]),
            generator=generator,
        )
        x_T = x_T.to(self.model.device)
        # DEBUG
        print(f"DEBUG PIPELINE: Calling sample with return_intermediate=True")
        try:
            sample_out = dpm_solver.sample(
                x=x_T,
                steps=num_inference_step,
                order=order,
                skip_type=skip_type,
                method=method,
                return_intermediate=True
            ) 
        except TypeError:
             # Fallback nếu file dpm_solver chưa sửa xong
            print("Warning: dpm_solver.sample chưa hỗ trợ return_intermediate. Đang chạy chế độ cũ.")
            sample_out = dpm_solver.sample(
                x=x_T, steps=num_inference_step, order=order, skip_type=skip_type, method=method
            )
        
        intermediates = []
        
        # Kiểm tra xem kết quả trả về là 1 biến hay 2 biến
        if isinstance(sample_out, tuple):
            x_sample, intermediates_raw = sample_out
        else:
            x_sample = sample_out
            intermediates_raw = []

        # Xử lý ảnh cuối cùng (Final Image)
        x_sample = (x_sample / 2 + 0.5).clamp(0, 1)
        x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()
        x_images = self.numpy_to_pil(x_sample)

        # Xử lý các ảnh trung gian (Intermediate Images) -> Để làm GIF
        intermediates_pil = []
        for x_t in intermediates_raw:
            # Quy trình y hệt ảnh cuối: Denormalize -> Permute -> Numpy -> PIL
            x_t = (x_t / 2 + 0.5).clamp(0, 1)
            x_t = x_t.cpu().permute(0, 2, 3, 1).numpy()
            # numpy_to_pil trả về list, ta lấy phần tử đầu tiên (batch=1)
            intermediates_pil.append(self.numpy_to_pil(x_t)[0])

        return x_images, intermediates_pil