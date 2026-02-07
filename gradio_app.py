import os
import glob
import random
import gradio as gr
from PIL import Image
import numpy as np
import cv2
from sample import (arg_parse, sampling, load_fontdiffuer_pipeline)

def fake_brush_stroke(pil_image, thickness=3, blur=3):
    if pil_image is None: 
        return None
    
    img_arr = np.array(pil_image.convert("L"))
    img_arr = 255 - img_arr
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    
    dilated = cv2.dilate(img_arr, kernel, iterations=1)
    
    if blur % 2 == 0: 
        blur += 1
    blurred = cv2.GaussianBlur(dilated, (blur, blur), 0)
    
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    result = 255 - thresh
    
    return Image.fromarray(result).convert("RGB")

def process_image(input_image, apply_brush=False, brush_thickness=3):
    if input_image is None:
        return None
   
    if isinstance(input_image, dict):
        input_image = input_image['composite']
        
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image

    img = img.convert("RGBA")
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    
    if apply_brush:
        background = fake_brush_stroke(background, thickness=brush_thickness)
        
    return background

def run_wrapper(mode, 
                upload_A, upload_B, 
                mix_upload_A, mix_draw_B,
                batch_upload_style, batch_draw_style,
                sampling_step, guidance_scale, batch_size):
    
    args.character_input = False 
    args.content_character = ""
    args.sampling_step = sampling_step
    args.guidance_scale = guidance_scale
    args.batch_size = batch_size
    args.seed = random.randint(0, 10000)
    args.rsi = False
    args.mca = False
    
    results = []

    if "Batch" in mode:
        content_folder = "vietnamese_glyphs"
        if not os.path.exists(content_folder):
            os.makedirs(content_folder, exist_ok=True)
            raise gr.Error(f"Folder '{content_folder}' not found. Please create it and add content images.")
            
        content_files = sorted(glob.glob(os.path.join(content_folder, "*.*")))
        if not content_files:
            raise gr.Error(f"No images found in '{content_folder}'.")

        if mode == "Batch: Upload Style":
            ref_raw = batch_upload_style
            apply_brush = False
        else: 
            ref_raw = batch_draw_style
            apply_brush = True
            
        if ref_raw is None:
            raise gr.Error("Missing Style image.")

        reference_image = process_image(ref_raw, apply_brush=apply_brush)
        
        for file_path in content_files:
            try:
                src_raw = Image.open(file_path)
                source_image = process_image(src_raw)
                
                out = sampling(args=args, pipe=pipe, content_image=source_image, style_image=reference_image)
                if out:
                    results.append(out.resize((512, 512)))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
                
        return results

    else:
        if mode == "Upload All":
            src_raw = upload_A
            ref_raw = upload_B
            apply_brush_ref = False
        else: # Mode Mix
            src_raw = mix_upload_A
            ref_raw = mix_draw_B
            apply_brush_ref = True

        if src_raw is None or ref_raw is None:
            raise gr.Error("Missing input images")

        source_image = process_image(src_raw, apply_brush=False) # Content khong can brush effect
        reference_image = process_image(ref_raw, apply_brush=apply_brush_ref)

        out_image = sampling(args=args, pipe=pipe, content_image=source_image, style_image=reference_image)
        
        if out_image is not None:
            results.append(out_image.resize((512, 512)))
            
        return results

def toggle_mode(mode):
    vis = [False] * 4
    
    if mode == "Upload All":
        vis[0] = True
    elif mode == "Mix":
        vis[1] = True
    elif mode == "Batch: Upload Style":
        vis[2] = True
    elif mode == "Batch: Draw Style":
        vis[3] = True
        
    return [gr.update(visible=v) for v in vis]

def update_brush_size(size):
    updates = [gr.update(brush_radius=size)] * 2
    return tuple(updates)

if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    if args.ckpt_dir is None:   
        args.ckpt_dir = '/kaggle/input/fontdiffuser-p2-same-both-64/p2_64_cross_both_bs4_neg04/global_step_30000'
    args.ttf_path = 'ttf/LXGWWenKaiTC-Bold.ttf'
    
    pipe = load_fontdiffuer_pipeline(args=args)

    css = ".output-gallery { height: 600px !important; } .output-gallery img { object-fit: contain; }"

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=css, title="FontDiffuser with CL-SCR") as demo:
        gr.Markdown("# FontDiffuser with CL-SCR")

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, variant="panel"):
                input_mode = gr.Radio(
                    choices=["Upload All", "Mix", "Batch: Upload Style", "Batch: Draw Style"], 
                    value="Upload All", 
                    label="Input Mode"
                )
                
                brush_slider = gr.Slider(label="Brush Size", minimum=1, maximum=50, value=10, step=1)

                with gr.Group(visible=True) as group_upload:
                    with gr.Row():
                        upload_A = gr.Image(label="Content (Upload)", type="pil", height=300)
                        upload_B = gr.Image(label="Style (Upload)", type="pil", height=300)

                with gr.Group(visible=False) as group_mix:
                      with gr.Row():
                          mix_upload_A = gr.Image(label="Content (Upload)", type="pil", height=350)
                          mix_draw_B = gr.Image(source="canvas", tool="sketch", type="pil", label="Style (Draw)", brush_radius=10, height=350)

                with gr.Group(visible=False) as group_batch_upload:
                    gr.Markdown("Auto-generate for images in folder `vietnamese_glyphs`")
                    batch_upload_style = gr.Image(label="Style Reference (Upload)", type="pil", height=350)

                with gr.Group(visible=False) as group_batch_draw:
                    gr.Markdown("Auto-generate for images in folder `vietnamese_glyphs`")
                    batch_draw_style = gr.Image(source="canvas", tool="sketch", type="pil", label="Style Reference (Draw)", brush_radius=10, height=350)

            with gr.Column(scale=2, variant="panel"):
                btn_run = gr.Button("Generate", variant="primary")
                
                output_gallery = gr.Gallery(label="Results", columns=2, height=600, elem_classes="output-gallery", object_fit="contain")
                
                with gr.Accordion("Settings", open=False):
                    sampling_step = gr.Slider(20, 50, value=20, step=10, label="Sampling Steps")
                    guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, label="Guidance Scale")
                    batch_size = gr.Slider(1, 2, value=1, step=1, label="Batch Size")

        input_mode.change(fn=toggle_mode, inputs=input_mode, outputs=[group_upload, group_mix, group_batch_upload, group_batch_draw])
        
        brush_slider.change(fn=update_brush_size, inputs=brush_slider, outputs=[mix_draw_B, batch_draw_style])
        
        btn_run.click(
            fn=run_wrapper, 
            inputs=[
                input_mode, 
                upload_A, upload_B, 
                mix_upload_A, mix_draw_B,
                batch_upload_style, batch_draw_style,
                sampling_step, guidance_scale, batch_size
            ], 
            outputs=output_gallery
        )
    
    demo.launch(debug=True)