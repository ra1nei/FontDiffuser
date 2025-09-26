python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/え.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/851CHIKARA-DZUYOKU-kanaB-2_english/851CHIKARA-DZUYOKU-kanaB-2_english+t.png" \
    --save_image \
    --save_image_dir="./hans-eng/example1" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/事.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/A-OTF-CinemaLetterStd-Light-2_english/A-OTF-CinemaLetterStd-Light-2_english+D+.png" \
    --save_image \
    --save_image_dir="./hans-eng/example2" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/专.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/Classic Zong yi ti Font_english/Classic Zong yi ti Font_english+U+.png" \
    --save_image \
    --save_image_dir="./hans-eng/example3" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
