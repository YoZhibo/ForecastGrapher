if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/traffic" ]; then
    mkdir ./logs/traffic
fi
export CUDA_VISIBLE_DEVICES=0

seq_len=96
model_name=ForecastGrapher

for k in 4
do
  pred_len=96
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --k $k \
    --z 8 \
    --node_dim 1000 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 >logs/traffic/$model_name'_'traffic_$seq_len'_'$pred_len.log

  pred_len=192
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --k $k \
    --z 8 \
    --node_dim 1000 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 >logs/traffic/$model_name'_'traffic_$seq_len'_'$pred_len.log

  pred_len=336
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --k $k \
    --z 8 \
    --node_dim 1000 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 >logs/traffic/$model_name'_'traffic_$seq_len'_'$pred_len.log

  pred_len=720
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path traffic.csv \
    --model_id traffic'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --k $k \
    --z 8 \
    --node_dim 1000 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --itr 1 >logs/traffic/$model_name'_'traffic_$seq_len'_'$pred_len.log
done