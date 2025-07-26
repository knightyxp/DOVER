export CUDA_VISIBLE_DEVICES=0,1,2,3
python evaluate_a_set_of_videos.py \
    -in /scratch3/yan204/yxp/Senorita/local_style_transfer_upload \
    -out /scratch3/yan204/yxp/DOVER/senorita_predictions/local_style_transfer_dover_score.csv \
    -d cuda