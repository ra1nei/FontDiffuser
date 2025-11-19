git pull

#######
python new_inference.py \
  --ckpt_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p1-same - 96x96"  \
  --source_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source" \
  --english_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english" \
  --chinese_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese" \
  --random_style \
  --save_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_same"

python new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_same" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_same.txt"

#######
python new_inference.py \
  --ckpt_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p1-cross - 96x96"  \
  --source_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source" \
  --english_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english" \
  --chinese_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese" \
  --random_style \
  --save_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_cross"

python new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_cross" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p1_cross.txt"

#######
python new_inference.py \
  --ckpt_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2-same-both - 96x96"  \
  --source_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source" \
  --english_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english" \
  --chinese_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese" \
  --random_style \
  --save_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_same_both"

python new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_same_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_same_both.txt"

#######
python new_inference.py \
  --ckpt_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2-cross-both - 96x96"  \
  --source_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source" \
  --english_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english" \
  --chinese_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese" \
  --random_style \
  --save_dir "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_cross_both"

python new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_cross_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/results/96->64_random/SFUC_p2_cross_both.txt"