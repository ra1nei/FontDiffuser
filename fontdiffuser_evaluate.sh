#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

conda activate fontdiffuser

echo ">>> Đang chạy trong môi trường: $(conda info --envs | grep '*' | awk '{print $1}')"

git pull && python fontdiffuser_evaluate.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-same \
  --source_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/chinese \
  --name p1_same_SFUC

git pull && python fontdiffuser_evaluate.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-cross \
  --source_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/chinese \
  --name p1_cross_SFUC

git pull && python fontdiffuser_evaluate.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-same-both \
  --source_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/chinese \
  --name p2_same_both_SFUC

git pull && python fontdiffuser_evaluate.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-cross-both \
  --source_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/source \
  --english_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/english \
  --chinese_dir /datastore/cndt_hangdv/TDKD/font_translator_gan/datasets/test_unknown_content/chinese \
  --name p2_cross_both_SFUC

# conda deactivate