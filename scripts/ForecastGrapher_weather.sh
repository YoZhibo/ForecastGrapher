if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/weather" ]; then
    mkdir ./logs/weather
fi
export CUDA_VISIBLE_DEVICES=1


seq_len=96
model_name=ForecastGrapher

for k in 3
do
  pred_len=96
  python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path weather.csv \
      --model_id weather'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --freq h \
      --target 'OT' \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --factor 3 \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des 'Exp' \
      --d_model 512 \
      --k $k \
      --z 32 \
      --batch_size 32 \
      --itr 1 >logs/weather/$model_name'_'weather_$seq_len'_'$pred_len.log

  pred_len=192
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path weather.csv \
     --model_id weather'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --factor 3 \
     --enc_in 21 \
     --dec_in 21 \
     --c_out 21 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 32 \
     --batch_size 32 \
     --itr 1  >logs/weather/$model_name'_'weather_$seq_len'_'$pred_len.log
#
  pred_len=336
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path weather.csv \
     --model_id weather'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --factor 3 \
     --enc_in 21 \
     --dec_in 21 \
     --c_out 21 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 32 \
     --batch_size 32 \
     --itr 1 >logs/weather/$model_name'_'weather_$seq_len'_'$pred_len.log
#
  pred_len=720
  python -u run.py \
     --is_training 1 \
     --root_path ./dataset/ \
     --data_path weather.csv \
     --model_id weather'_'$seq_len'_'$pred_len \
     --model $model_name \
     --data custom \
     --features M \
     --freq h \
     --target 'OT' \
     --seq_len $seq_len \
     --pred_len $pred_len \
     --e_layers 2 \
     --d_layers 1 \
     --factor 3 \
     --enc_in 21 \
     --dec_in 21 \
     --c_out 21 \
     --des 'Exp' \
     --d_model 512 \
     --k $k \
     --z 32 \
     --batch_size 32 \
     --itr 1 >logs/weather/$model_name'_'weather_$seq_len'_'$pred_len.log
done