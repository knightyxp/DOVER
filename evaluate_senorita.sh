torchrun --nproc_per_node=4 \
    evaluate_a_set_of_videos_ddp.py \
    -in "/scratch3/yan204/yxp/Senorita/local_style_transfer_upload" \
    -out "./senorita_predictions/local_style_transfer_dover_score.csv" \
    -bs 4