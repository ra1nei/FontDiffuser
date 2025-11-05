####### SFUC #######
source_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_content/source"
english_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_content/english"
chinese_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_content/chinese"
name="SFUC"

# ### P1 Same
# model="p1_same"
# git pull && python fontdiffuser_inference.py \
#   --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-same \
#   --source_dir $source_dir \
#   --english_dir $english_dir \
#   --chinese_dir $chinese_dir \
#   --name $name \
#   --model $model && \
# python fontdiffuser_evaluate.py \
#   --name $name \
#   --model $model


# ### P1 Cross
# model="p1_cross"
# git pull && python fontdiffuser_inference.py \
#   --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-cross \
#   --source_dir $source_dir \
#   --english_dir $english_dir \
#   --chinese_dir $chinese_dir \
#   --name $name \
#   --model $model && \
# python fontdiffuser_evaluate.py \
#   --name $name \
#   --model $model

# ### P2 Same Both
# model="p2_same_both"
# git pull && python fontdiffuser_inference.py \
#   --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-same-both \
#   --source_dir $source_dir \
#   --english_dir $english_dir \
#   --chinese_dir $chinese_dir \
#   --name $name \
#   --model $model && \
# python fontdiffuser_evaluate.py \
#   --name $name \
#   --model $model


### P2 Cross Both
model="p2_cross_both"
git pull && python fontdiffuser_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-cross-both \
  --source_dir $source_dir \
  --english_dir $english_dir \
  --chinese_dir $chinese_dir \
  --name $name \
  --model $model && \
python fontdiffuser_evaluate.py \
  --name $name \
  --model $model












####### UFSC #######
source_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_style/source"
english_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_style/english"
chinese_dir="/datastore/cndt_hangdv/TDKD/FCAGAN/font_translator_gan/test_unknown_style/chinese"
name="UFSC"

### P1 Same
model="p1_same"
git pull && python fontdiffuser_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-same \
  --source_dir $source_dir \
  --english_dir $english_dir \
  --chinese_dir $chinese_dir \
  --name $name \
  --model $model && \
python fontdiffuser_evaluate.py \
  --name $name \
  --model $model


### P1 Cross
model="p1_cross"
git pull && python fontdiffuser_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p1-cross \
  --source_dir $source_dir \
  --english_dir $english_dir \
  --chinese_dir $chinese_dir \
  --name $name \
  --model $model && \
python fontdiffuser_evaluate.py \
  --name $name \
  --model $model

### P2 Same Both
model="p2_same_both"
git pull && python fontdiffuser_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-same-both \
  --source_dir $source_dir \
  --english_dir $english_dir \
  --chinese_dir $chinese_dir \
  --name $name \
  --model $model && \
python fontdiffuser_evaluate.py \
  --name $name \
  --model $model


### P2 Cross Both
model="p2_cross_both"
git pull && python fontdiffuser_inference.py \
  --ckpt_dir /datastore/cndt_hangdv/TDKD/FontDiffuser/ckpt/p2-cross-both \
  --source_dir $source_dir \
  --english_dir $english_dir \
  --chinese_dir $chinese_dir \
  --name $name \
  --model $model && \
python fontdiffuser_evaluate.py \
  --name $name \
  --model $model