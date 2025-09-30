python sample.py \
    --ckpt_dir="./p2_unet_same_both/global_step_30000/" \
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
    --ckpt_dir="./p2_unet_same_both/global_step_30000/" \
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
    --ckpt_dir="./p2_unet_same_both/global_step_30000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/え.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/Classic Zong yi ti Font_chinese/Classic Zong yi ti Font_chinese+专.png" \
    --save_image \
    --save_image_dir="./hans-hans/example3" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
