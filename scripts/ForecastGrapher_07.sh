if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/PEMS07" ]; then
    mkdir ./logs/PEMS07
fi

export CUDA_VISIBLE_DEVICES=2

seq_len=96
model_name=ForecastGrapher

for k in 3
do
  pred_len=12
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS07/ \
        --data_path PEMS07.npz \
        --model_id PEMS07_96_12 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 883 \
        --dec_in 883 \
        --c_out 883 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.001 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS07/$model_name'_'PEMS07_$seq_len'_'$pred_len.log

  pred_len=24
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS07/ \
        --data_path PEMS07.npz \
        --model_id PEMS07_96_24 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 883 \
        --dec_in 883 \
        --c_out 883 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.001 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS07/$model_name'_'PEMS07_$seq_len'_'$pred_len.log

  pred_len=48
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS07/ \
        --data_path PEMS07.npz \
        --model_id PEMS07_96_48 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 883 \
        --dec_in 883 \
        --c_out 883 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.001 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS07/$model_name'_'PEMS07_$seq_len'_'$pred_len.log

  pred_len=96
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS07/ \
        --data_path PEMS07.npz \
        --model_id PEMS07_96_96 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 2 \
        --factor 3 \
        --enc_in 883 \
        --dec_in 883 \
        --c_out 883 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.001 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS07/$model_name'_'PEMS07_$seq_len'_'$pred_len.log
done