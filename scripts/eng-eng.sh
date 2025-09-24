python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/A.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/AaChaoMianJin-2_english/AaChaoMianJin-2_english+L+.png" \
    --save_image \
    --save_image_dir="./eng-eng/example1" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/A.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/║║╥╟▓╩╡√╠σ╝≥_english/║║╥╟▓╩╡√╠σ╝≥_english+M+.png" \
    --save_image \
    --save_image_dir="./eng-eng/example2" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"

python sample.py \
    --ckpt_dir="./global_step_440000/" \
    --content_image_path="./thesis-data-png/train/ContentImage/A.png" \
    --style_image_path="./thesis-data-png/train/TargetImage/851CHIKARA-DZUYOKU-kanaB-2_english/851CHIKARA-DZUYOKU-kanaB-2_english+b.png" \
    --save_image \
    --save_image_dir="./eng-eng/example3" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
