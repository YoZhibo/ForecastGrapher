if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/PEMS03" ]; then
    mkdir ./logs/PEMS03
fi

export CUDA_VISIBLE_DEVICES=0

seq_len=96
model_name=ForecastGrapher

for k in 3
do
  pred_len=12
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS03/ \
        --data_path PEMS03.npz \
        --model_id PEMS03_96_12 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --use_norm 0 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --itr 1 >logs/PEMS03/$model_name'_'PEMS03_$seq_len'_'$pred_len.log

  pred_len=24
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS03/ \
        --data_path PEMS03.npz \
        --model_id PEMS03_96_24 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --use_norm 0 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --itr 1 >logs/PEMS03/$model_name'_'PEMS03_$seq_len'_'$pred_len.log

  pred_len=48
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS03/ \
        --data_path PEMS03.npz \
        --model_id PEMS03_96_48 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --use_norm 0 \
        --learning_rate 0.001 \
        --batch_size 32 \
        --itr 1 >logs/PEMS03/$model_name'_'PEMS03_$seq_len'_'$pred_len.log

  pred_len=96
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS03/ \
        --data_path PEMS03.npz \
        --model_id PEMS03_96_96 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 358 \
        --dec_in 358 \
        --c_out 358 \
        --des 'Exp' \
        --d_model 1024 \
        --z 8 \
        --k $k \
        --batch_size 32 \
        --learning_rate 0.001 \
        --use_norm 0 \
        --itr 1 >logs/PEMS03/$model_name'_'PEMS03_$seq_len'_'$pred_len.log
done