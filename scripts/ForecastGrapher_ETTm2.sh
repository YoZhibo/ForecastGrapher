if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/ETTm2" ]; then
    mkdir ./logs/ETTm2
fi
export CUDA_VISIBLE_DEVICES=3

seq_len=96
model_name=ForecastGrapher

for k in 1
do
  pred_len=96
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 32 \
      --batch_size 32 \
      --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log

  pred_len=192
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 32 \
      --batch_size 32 \
      --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log


  pred_len=336
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 32 \
      --batch_size 32 \
      --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log


  pred_len=720
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm2.csv \
      --model_id ETTm2'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data ETTm2 \
      --features M \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 32 \
      --batch_size 32 \
      --itr 1 >logs/ETTm2/$model_name'_'ETTm2_$seq_len'_'$pred_len.log
done