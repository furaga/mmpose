#!/bin/bash

for n in tokusha yorimoi-op sunshine-op2 sunshine-op1 starlight8-12; do
python demo/bottom_up_img_auto_annotation.py  \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/crowdpose/higherhrnet_w32_anim_512x512_udp.py \
    checkpoints/epoch_130-20210925.pth  \
    --img_dir /mnt/d/workspace/anim/screenshots/RAW/${n} \
    --out_dir /mnt/d/workspace/anim/screenshots/posed/${n} 
done
