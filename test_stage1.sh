#!/bin/bash
python model_zalando_mask_content_test.py \
  --checkpoint model/stage1/model-15000 \
  --mode test \
  --result_dir results/stage1_1/ \
  --begin 0 \
  --end 15
