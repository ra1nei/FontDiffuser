##### SFUC #####

### random ###
python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2_same_both_64/global_step_30000 \
  --source_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese \
  --random_style \
  --save_dir '/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_same_both'

python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_same_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_same_both.txt"




python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2_cross_both_64/global_step_30000 \
  --source_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese \
  --random_style \
  --save_dir '/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_cross_both'

python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_cross_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_random/SFUC/p2_cross_both.txt"


### a ###
python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2_same_both_64/global_step_30000 \
  --source_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese \
  --save_dir '/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_same_both'

python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_same_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_same_both.txt"




python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/checkpoint/p2_cross_both_64/global_step_30000 \
  --source_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/image_dataset/FTransGAN/test_unknown_content/chinese \
  --save_dir '/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_cross_both'

python /datastore/cndt_hangdv/TDKD/FontDiffuser/new_evaluation.py "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_cross_both" \
  --output "/datastore/cndt_hangdv/TDKD/FontDiffuser/inference/64_a/SFUC/p2_cross_both.txt"