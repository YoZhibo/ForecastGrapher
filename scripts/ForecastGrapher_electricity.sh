if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/electricity" ]; then
    mkdir ./logs/electricity
fi
export CUDA_VISIBLE_DEVICES=3

seq_len=96
model_name=ForecastGrapher

for k in 4
do
  pred_len=96
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path electricity.csv \
      --model_id electricity'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --freq h \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 8 \
      --learning_rate 0.0005 \
      --batch_size 32 \
      --itr 1 >logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log

  pred_len=192
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path electricity.csv \
     --model_id electricity'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --factor 3 \
     --enc_in 321 \
     --dec_in 321 \
     --c_out 321 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 8 \
     --batch_size 32 \
     --learning_rate 0.0005 \
     --itr 1 >logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log


  pred_len=336
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path electricity.csv \
     --model_id electricity'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --factor 3 \
     --enc_in 321 \
     --dec_in 321 \
     --c_out 321 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 8 \
     --batch_size 32 \
     --learning_rate 0.0005 \
     --itr 1 >logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log


  pred_len=720
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path electricity.csv \
     --model_id electricity'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --factor 3 \
     --enc_in 321 \
     --dec_in 321 \
     --c_out 321 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 8 \
     --batch_size 32 \
     --learning_rate 0.0005 \
     --itr 1 >logs/electricity/$model_name'_'electricity_$seq_len'_'$pred_len.log

done