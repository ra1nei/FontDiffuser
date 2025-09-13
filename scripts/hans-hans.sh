python sample.py \
    --ckpt_dir="./outputs/p1-unet-same/global_step_15000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/え.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/851CHIKARA-DZUYOKU-kanaB-2_chinese/851CHIKARA-DZUYOKU-kanaB-2_chinese+万.png" \
    --save_image \
    --save_image_dir="./hans-hans/example1" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./outputs/p1-unet-same/global_step_15000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/え.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/A-OTF-CinemaLetterStd-Light-2_chinese/A-OTF-CinemaLetterStd-Light-2_chinese+事.png" \
    --save_image \
    --save_image_dir="./hans-hans/example2" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./outputs/p1-unet-same/global_step_15000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/え.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/Arphic ‘Who’ E5GBK_M Handwriting Pen Chinese Font-Simplified Chinese Fonts_chinese/Arphic ‘Who’ E5GBK_M Handwriting Pen Chinese Font-Simplified Chinese Fonts_chinese+且.png" \
    --save_image \
    --save_image_dir="./hans-hans/example3" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
