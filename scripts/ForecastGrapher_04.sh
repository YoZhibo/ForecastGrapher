if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/PEMS04" ]; then
    mkdir ./logs/PEMS04
fi

export CUDA_VISIBLE_DEVICES=1

seq_len=96
model_name=ForecastGrapher

for k in 3
do
  pred_len=12
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS04/ \
        --data_path PEMS04.npz \
        --model_id PEMS04_96_12 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.0005 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS04/$model_name'_'PEMS04_$seq_len'_'$pred_len.log
#
  pred_len=24
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS04/ \
        --data_path PEMS04.npz \
        --model_id PEMS04_96_24 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.0005 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS04/$model_name'_'PEMS04_$seq_len'_'$pred_len.log

  pred_len=48
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS04/ \
        --data_path PEMS04.npz \
        --model_id PEMS04_96_48 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.0005 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS04/$model_name'_'PEMS04_$seq_len'_'$pred_len.log

  pred_len=96
  python -u run.py \
        --is_training 1 \
        --root_path ./dataset/PEMS04/ \
        --data_path PEMS04.npz \
        --model_id PEMS04_96_96 \
        --model $model_name \
        --data PEMS \
        --features M \
        --freq h \
        --target 'OT' \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 3 \
        --factor 3 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --des 'Exp' \
        --d_model 512 \
        --z 8 \
        --k $k \
        --learning_rate 0.0005 \
        --use_norm 0 \
        --batch_size 32 \
        --itr 1 >logs/PEMS04/$model_name'_'PEMS04_$seq_len'_'$pred_len.log
done